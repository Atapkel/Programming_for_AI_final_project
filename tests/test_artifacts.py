import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src import artifacts as a, data_loader as loader, database as db
from src import dashboard_data as ui


def test_invalid_replacement_preserves_previous_bytes(tmp_path):
    dest = tmp_path / "data.csv"; dest.write_text("good data")
    def reject(path): raise ValueError("invalid schema")
    with pytest.raises(ValueError):
        with a.atomic_path(dest, reject) as temp:
            temp.write_text("bad data")
    assert dest.read_text() == "good data"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data.csv"]


def test_partial_bundle_failure_rolls_back_all_replacements(tmp_path, monkeypatch):
    dest = tmp_path / "dest"; dest.mkdir()
    stage = tmp_path / "stage"; stage.mkdir()
    for name in ("a", "b"):
        (dest / name).write_text("old " + name)
        (stage / name).write_text("new " + name)
    original = a.os.replace
    failed = False
    def fail_once(src, dst):
        nonlocal failed
        if Path(dst) == dest / "b" and not failed:
            failed = True
            raise OSError("simulated disk error")
        return original(src, dst)
    monkeypatch.setattr(a.os, "replace", fail_once)
    with pytest.raises(OSError):
        a.publish_files({dest / n: stage / n for n in ("a", "b")}, dest)
    assert (dest / "a").read_text() == "old a"
    assert (dest / "b").read_text() == "old b"
    a.assert_idle(dest)


def test_empty_download_never_overwrites_existing_source(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "RAW_DIR", tmp_path)
    path = tmp_path / "usd_kzt_daily.csv"; path.write_text("date,rate\n2020-01-01,100\n")
    old = path.read_bytes()
    with pytest.raises(ValueError):
        loader.save_if_needed("usd_kzt_daily", lambda: pd.DataFrame(columns=["date", "rate"]), force=True)
    assert path.read_bytes() == old


def test_database_build_failure_keeps_old_database(tmp_path, monkeypatch):
    path = tmp_path / "macro.db"; path.write_bytes(b"existing database")
    def broken(temp):
        temp.write_bytes(b"partial database")
        raise ValueError("inconsistent processed inputs")
    monkeypatch.setattr(db, "_build_database", broken)
    with pytest.raises(ValueError):
        db.build_database(path)
    assert path.read_bytes() == b"existing database"


def test_manifest_detects_corruption(tmp_path):
    output = tmp_path / "result.json"; output.write_text('{"value": 1}')
    value = a.manifest("test", {}, {output.name: output})
    a.write_json(tmp_path / "manifest.json", value)
    assert a.verify_manifest(tmp_path, "manifest.json")["run_id"] == value["run_id"]
    output.write_text('{"value": 2}')
    with pytest.raises(a.ArtifactError, match="does not match"):
        a.verify_manifest(tmp_path, "manifest.json")


def test_missing_model_result_is_not_cached(tmp_path):
    assert ui.load_model_results(tmp_path) is None
    source = Path(__file__).resolve().parents[1] / "data" / "processed"
    # Legacy compatibility is deliberately tested, without a new manifest.
    for name in (*ui.MODEL_FILES, "monthly_macro.parquet"):
        shutil.copyfile(source / name, tmp_path / name)
    result = ui.load_model_results(tmp_path)
    assert result is not None and not result["metrics"].empty
    path = tmp_path / "model_metrics.csv"
    metrics = pd.read_csv(path); metrics.loc[0, "MAE"] += 123
    metrics.to_csv(path, index=False)
    assert ui.load_model_results(tmp_path)["metrics"].MAE.iloc[0] == pytest.approx(metrics.MAE.iloc[0])


def test_same_length_csv_change_invalidates_cache(tmp_path):
    path = tmp_path / "test.csv"; path.write_text("value\n100\n")
    assert ui.load_csv(path).value.iloc[0] == 100
    path.write_text("value\n200\n")
    assert ui.load_csv(path).value.iloc[0] == 200


def test_interrupted_publication_is_visible_to_reader(tmp_path):
    (tmp_path / ".update-in-progress").write_text("interrupted")
    with pytest.raises(a.ArtifactError, match="interrupted"):
        ui.load_model_results(tmp_path)


def test_full_pipeline_failure_leaves_destination_untouched(tmp_path, monkeypatch):
    import main
    raw = tmp_path / "raw"; raw.mkdir(); (raw / "input.csv").write_text("value\n1\n")
    processed = tmp_path / "processed"; processed.mkdir()
    old = processed / "model_metrics.csv"; old.write_text("old metrics")
    monkeypatch.setattr(main, "RAW_DIR", raw)
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    calls = []
    def fail(*args, **kw):
        calls.append(args[0][-1])
        staged = Path(kw["env"]["MACRO_DATA_DIR"]) / "processed"
        staged.mkdir(exist_ok=True)
        (staged / "model_metrics.csv").write_text("new staged output")
        if len(calls) == 4:
            raise RuntimeError("training failure")
    monkeypatch.setattr(main.subprocess, "run", fail)
    with pytest.raises(RuntimeError):
        main.step_full_pipeline()
    assert old.read_text() == "old metrics"
    assert len(calls) == 4  # failure AFTER the preceding stages have completed


def test_batch_download_failure_does_not_publish_first_success(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "data" / "raw"
    raw = tmp_path / "raw"
    shutil.copytree(source, raw)
    previous = a.fingerprint(sorted(raw.glob("*.csv")))
    monkeypatch.setattr(loader, "RAW_DIR", raw)
    fresh = pd.read_csv(raw / "usd_kzt_daily.csv")
    fresh.loc[0, "rate"] += 1
    monkeypatch.setattr(loader, "download_usd_kzt", lambda: fresh)
    monkeypatch.setattr(loader, "download_base_rate", lambda: pd.DataFrame())
    with pytest.raises(ValueError, match="preserved"):
        loader.download_all(force=True)
    assert a.fingerprint(sorted(raw.glob("*.csv"))) == previous


def test_model_bundle_rejects_missing_model_and_inconsistent_metrics(tmp_path):
    from src.models import validate_model_outputs
    source = Path(__file__).resolve().parents[1] / "data" / "processed"
    for name in (*ui.MODEL_FILES, "model_input.parquet"):
        shutil.copyfile(source / name, tmp_path / name)
    validate_model_outputs(tmp_path)
    path = tmp_path / "model_metrics.csv"
    scores = pd.read_csv(path); scores.loc[1, "RMSE"] += 10
    scores.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Metrics/fallback"):
        validate_model_outputs(tmp_path)
    shutil.copyfile(source / "model_metrics.csv", path)
    bt_path = tmp_path / "backtest_predictions.parquet"
    bt = pd.read_parquet(bt_path)
    bt[bt.model != "gradient_boosting"].to_parquet(bt_path, index=False)
    with pytest.raises(ValueError, match="metrics"):
        validate_model_outputs(tmp_path)


def test_sql_queries_execute_against_current_database():
    db.validate_database(db.DB_PATH)
    assert len(db.run_all_queries(save=False)) == 11
