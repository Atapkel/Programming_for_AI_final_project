"""Console menu for the Kazakhstan macro project.

Run:
    python main.py
"""

import subprocess
import sys
import time
from pathlib import Path

import requests

from src import analysis, data_loader, database, models, preprocessing

ROOT = Path(__file__).resolve().parent


def step_download():
    force = ask_yes_no("Download again even if files exist?")
    data_loader.download_all(force=force)


def step_preprocess():
    tables, report = preprocessing.run_preprocessing()
    for name, table in tables.items():
        print(f"  {name:18s} {len(table):6d} rows")
    print(f"  USD/KZT days filled by interpolation: {report['usd_kzt_filled_days']}")


def step_database():
    database.build_database()
    database.run_all_queries()


def step_analysis():
    results = analysis.run_analysis()
    for name, r in results["stationarity"].items():
        print(f"  {name:22s} ADF p={r['adf_p']:.3f}  KPSS p={r['kpss_p']:.3f}  -> {r['verdict']}")
    print(f"  Figures saved to {analysis.FIGURES_DIR}")


def step_models():
    print("  Backtesting models (about 1 minute)...")
    summary = models.run_models()
    for target, info in summary.items():
        print(f"  {target}: best model = {info['best_model']}")


def step_show_query():
    names = list(database.QUERIES)
    for i, name in enumerate(names, start=1):
        print(f"  {i:2d}. {database.QUERIES[name][0]}")
    choice = ask_int("Query number", 1, len(names))
    result = database.run_query(names[choice - 1])
    print(result.head(20).to_string(index=False))
    if len(result) > 20:
        print(f"  ... {len(result) - 20} more rows")


def step_dashboard():
    print("  Starting Streamlit - press Ctrl+C to stop and return to the menu")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py")], check=False)
    except KeyboardInterrupt:
        print("\n  Dashboard stopped")


def step_full_pipeline():
    for title, func in PIPELINE:
        print(f"\n--- {title} ---")
        func()


PIPELINE = [
    ("Preprocess data", step_preprocess),
    ("Build database and run SQL", step_database),
    ("Time-series analysis", step_analysis),
    ("Train and evaluate models", step_models),
]

MENU = {
    "1": ("Download data", step_download),
    "2": ("Preprocess data", step_preprocess),
    "3": ("Build database and run SQL queries", step_database),
    "4": ("Time-series analysis and figures", step_analysis),
    "5": ("Train and evaluate forecasting models", step_models),
    "6": ("Run full pipeline (steps 2-5)", step_full_pipeline),
    "7": ("Show an SQL query result", step_show_query),
    "8": ("Open dashboard", step_dashboard),
    "0": ("Exit", None),
}


def ask_yes_no(question):
    while True:
        answer = input(f"  {question} [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no", ""):
            return False
        print("  Please answer y or n")


def ask_int(prompt, low, high):
    while True:
        try:
            value = int(input(f"  {prompt} ({low}-{high}): "))
        except ValueError:
            print("  Please enter a number")
            continue
        if low <= value <= high:
            return value
        print(f"  Number must be between {low} and {high}")


def main():
    print("Kazakhstan macroeconomic indicators and exchange rate")
    while True:
        print("\n" + "\n".join(f"  {key}. {title}" for key, (title, _) in MENU.items()))
        choice = input("Choose an option: ").strip()
        if choice not in MENU:
            print("  Unknown option, try again")
            continue
        title, action = MENU[choice]
        if action is None:
            print("Bye!")
            break
        start = time.perf_counter()
        try:
            action()
        except FileNotFoundError as err:
            print(f"  [error] {err}\n  Run the earlier steps first (download -> preprocess -> database).")
        except requests.RequestException as err:
            print(f"  [network error] {err}")
        except KeyboardInterrupt:
            print("\n  Interrupted")
        else:
            print(f"  Done: {title} ({time.perf_counter() - start:.1f}s)")


if __name__ == "__main__":
    main()
