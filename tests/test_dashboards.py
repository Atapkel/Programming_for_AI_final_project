from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ['Overview', 'Data Explorer', 'Data Quality', 'Advanced Data Analysis',
         'Interactive Visualization', 'SQL Explorer', 'Model Evaluation', 'Future Forecasting']


@pytest.mark.parametrize('entry', ['app.py', 'app/main.py'])
def test_unified_navigation_and_compatibility(entry, monkeypatch):
    from src import evaluation, models
    def forbidden(*args, **kwargs):
        raise AssertionError('Navigation attempted training')
    monkeypatch.setattr(evaluation, 'run_evaluation', forbidden)
    monkeypatch.setattr(models, 'run_models', forbidden)
    monkeypatch.setattr(models, 'fit_sarima', forbidden)
    app = AppTest.from_file(str(ROOT / entry), default_timeout=30).run()
    assert app.sidebar.radio[0].options == PAGES
    for page in PAGES:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
        assert not app.error, page
    app.selectbox(key='future_target').set_value('usd_kzt').run()
    app.multiselect[0].set_value(['lag_regression', 'gradient_boosting']).run()
    assert not app.exception
    assert any('Last observed target' in item.value for item in app.caption)
    assert not any('unavailable' in item.value for item in app.warning)
    app.multiselect[0].set_value([]).run()
    assert not app.exception


def test_original_reference_remains_accessible():
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    app.sidebar.radio[0].set_value('Model Evaluation').run()
    next(w for w in app.radio if w.label == 'Experiment').set_value('Stage 1 · original one-step reference').run()
    app.get('segmented_control')[0].set_value('USD/KZT exchange rate').run()
    assert not app.exception
    assert any('Last observed target' in item.value for item in app.caption)
    assert any('future path is unavailable' in item.value for item in app.info)
    for selection in app.multiselect:
        selection.set_value([])
    app.run()
    assert not app.exception


def test_evaluation_horizons_and_window_filters():
    app = AppTest.from_file(str(ROOT / 'app/main.py'), default_timeout=30).run()
    app.sidebar.radio[0].set_value('Model Evaluation').run()
    app.selectbox(key='evaluation_target').set_value('usd_kzt')
    app.selectbox(key='evaluation_window').set_value('rolling120')
    app.select_slider[0].set_value(12).run()
    assert not app.exception
    assert any('target months:' in item.value for item in app.caption)
    table = app.dataframe[0].value
    assert len(table) == 6 and {'MAE', 'RMSE', 'sMAPE', 'MASE'} <= set(table)


def test_world_bank_requires_indicator_before_statistics():
    app = AppTest.from_file(str(ROOT / 'app/main.py'), default_timeout=30).run()
    app.sidebar.radio[0].set_value('Data Explorer').run()
    app.selectbox[0].set_value('World Bank indicators').run()
    assert not app.exception
    assert len(app.dataframe) == 0
    app.selectbox[1].set_value(app.selectbox[1].options[0]).run()
    assert not app.exception
    assert len(app.dataframe) > 0


def test_missing_sql_database_has_actionable_error_without_creating_file(tmp_path, monkeypatch):
    from src import database
    missing = tmp_path / 'missing.db'
    monkeypatch.setattr(database, 'DB_PATH', missing)
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    app.sidebar.radio[0].set_value('SQL Explorer').run()
    assert not app.exception
    assert any('SQL data unavailable' in item.value for item in app.error)
    assert not missing.exists()


def test_missing_stage2_artifacts_show_rebuild_command(monkeypatch, tmp_path):
    from src import dashboard_data
    monkeypatch.setattr(dashboard_data, 'PROCESSED_DIR', tmp_path)
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    app.sidebar.radio[0].set_value('Future Forecasting').run()
    assert not app.exception
    assert any('python -m src.evaluation' in item.value for item in app.info)
