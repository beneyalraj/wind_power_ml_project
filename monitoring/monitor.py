"""
monitoring/monitor.py — Evidently AI Drift Detection

WHAT THIS DOES:
  Compares recent predictions (current data) against training data
  distribution (reference data) to detect drift.

  Drift = the statistical distribution of inputs or predictions has
  shifted significantly from what the model was trained on.
  When drift occurs, model accuracy degrades silently — predictions
  keep returning 200 OK but are increasingly wrong.

WHAT IT MONITORS:
  1. Data drift    — have input features shifted?
  2. Target drift  — has prediction distribution changed?

HOW TO RUN:
  python monitoring/monitor.py

  Report saved to: reports/drift_report.html
  Accessible at:   http://13.219.46.116/report

HOW OFTEN TO RUN:
  - Automatically via cron on EC2 (daily at 2am)
  - Manually any time: docker-compose run --rm monitor

INSTALL:
  pip install evidently
"""

import sys
import json
import logging
import collections
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REFERENCE_PATH  = Path("monitoring/reference_data.csv")
PREDICTIONS_LOG = Path("logs/predictions.jsonl")
REPORTS_DIR     = Path("reports")
REPORT_PATH     = REPORTS_DIR / "drift_report.html"

# Minimum predictions needed before drift detection is meaningful
MIN_PREDICTIONS = 50

# Maximum lines to load from the prediction log.
# deque(maxlen=10000) automatically discards old lines, keeping only
# the newest 10,000 predictions. Prevents memory issues on long-running
# production systems where the log can grow to millions of lines.
MAX_LOG_LINES = 10_000


# -----------------------------------------------------------------------------
# Load Data
# -----------------------------------------------------------------------------
def load_reference() -> pd.DataFrame:
    """Load training data baseline — what the model was trained on."""
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Reference data missing at {REFERENCE_PATH}. "
            "Run the reference data generation script first."
        )
    df = pd.read_csv(REFERENCE_PATH)
    logger.info(f"✅ Reference data loaded: {len(df)} rows")
    return df


def load_current() -> pd.DataFrame:
    """
    Load the most recent predictions from the JSONL log.

    WHY deque(maxlen=10000):
      A plain list would load the entire file into memory.
      On a long-running production system the log could grow to
      millions of lines — crashing the monitor with OOM.
      deque with maxlen automatically discards old lines,
      keeping only the newest 10,000 predictions in memory.
      This also means drift is evaluated on RECENT behaviour,
      not the entire history — which is what you actually want.
    """
    if not PREDICTIONS_LOG.exists():
        raise FileNotFoundError(
            f"Predictions log missing at {PREDICTIONS_LOG}. "
            "Make sure the API has received at least one request."
        )

    records = []
    with open(PREDICTIONS_LOG, "r") as f:
        # Only keep the newest MAX_LOG_LINES lines
        tail_lines = collections.deque(f, maxlen=MAX_LOG_LINES)

        for line in tail_lines:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if len(records) == 0:
        raise ValueError("Predictions log is empty — no predictions logged yet.")

    df = pd.DataFrame(records)

    # Drop non-feature columns before drift analysis
    df = df.drop(columns=["timestamp", "model_version"], errors="ignore")

    logger.info(f"✅ Current predictions loaded: {len(df)} rows "
                f"(capped at last {MAX_LOG_LINES})")
    return df


# -----------------------------------------------------------------------------
# Run Drift Detection
# -----------------------------------------------------------------------------
def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> None:
    """
    Generates an Evidently HTML drift report comparing current
    predictions against the training data reference baseline.

    WHY ColumnMapping:
      Without it, Evidently guesses which columns are features and
      which is the prediction target. It might misclassify prediction_kw
      as a regular feature, giving misleading drift results.
      ColumnMapping explicitly tells Evidently the role of each column
      so statistical tests are applied correctly.

    TWO REPORTS IN ONE:
      1. DataDriftPreset   — checks each feature for distribution shift
                             using statistical tests (KS test for numerical)
      2. DataQualityPreset — checks for missing values, out-of-range values,
                             and data quality issues in current predictions
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently import ColumnMapping
    except ImportError:
        logger.error("Evidently not installed. Run: pip install evidently")
        sys.exit(1)

    # Use only columns present in both datasets
    shared_columns = [c for c in reference.columns if c in current.columns]
    logger.info(f"Comparing {len(shared_columns)} columns: {shared_columns}")

    reference_clean = reference[shared_columns].copy()
    current_clean   = current[shared_columns].copy()

    # Explicitly define column roles so Evidently applies correct tests
    column_mapping = ColumnMapping()
    column_mapping.prediction = "prediction_kw"
    column_mapping.numerical_features = [
        c for c in shared_columns if c != "prediction_kw"
    ]

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])

    report.run(
        reference_data=reference_clean,
        current_data=current_clean,
        column_mapping=column_mapping
    )

    REPORTS_DIR.mkdir(exist_ok=True)
    report.save_html(str(REPORT_PATH))
    logger.info(f"✅ Drift report saved: {REPORT_PATH}")


# -----------------------------------------------------------------------------
# Summary to Terminal
# -----------------------------------------------------------------------------
def print_summary(reference: pd.DataFrame, current: pd.DataFrame) -> None:
    """Prints a quick drift summary to terminal."""

    print("\n" + "="*60)
    print("DRIFT MONITORING SUMMARY")
    print("="*60)
    print(f"Reference rows : {len(reference)}")
    print(f"Current rows   : {len(current)}")
    print()

    feature_cols = [
        c for c in reference.columns
        if c in current.columns and c != "prediction_kw"
    ]

    print(f"{'Feature':<25} {'Ref Mean':>12} {'Cur Mean':>12} {'Δ%':>8}")
    print("-"*60)

    for col in feature_cols:
        ref_mean = reference[col].mean()
        cur_mean = current[col].mean()
        if ref_mean != 0:
            delta_pct = ((cur_mean - ref_mean) / abs(ref_mean)) * 100
            flag = " ⚠️" if abs(delta_pct) > 20 else ""
            print(
                f"{col:<25} {ref_mean:>12.3f} "
                f"{cur_mean:>12.3f} {delta_pct:>7.1f}%{flag}"
            )

    if "prediction_kw" in reference.columns and "prediction_kw" in current.columns:
        ref_pred = reference["prediction_kw"].mean()
        cur_pred = current["prediction_kw"].mean()
        if ref_pred != 0:
            pred_delta = ((cur_pred - ref_pred) / abs(ref_pred)) * 100
            flag = " ⚠️" if abs(pred_delta) > 20 else ""
            print(
                f"\n{'prediction_kw':<25} {ref_pred:>12.0f} "
                f"{cur_pred:>12.0f} {pred_delta:>7.1f}%{flag}"
            )

    print("="*60)
    print("⚠️  = mean shifted >20% from reference — review the full report")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    logger.info("Starting drift monitoring...")

    try:
        reference = load_reference()
        current   = load_current()
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    if len(current) < MIN_PREDICTIONS:
        logger.warning(
            f"Only {len(current)} predictions logged. "
            f"Drift detection needs at least {MIN_PREDICTIONS} for reliable results. "
            f"Continuing anyway..."
        )

    print_summary(reference, current)
    run_drift_report(reference, current)

    # Log URL — running on EC2, no browser available
    logger.info("✅ Monitoring complete.")
    logger.info("📊 Report accessible at: http://13.219.46.116/report")


if __name__ == "__main__":
    main()