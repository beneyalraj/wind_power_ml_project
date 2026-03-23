"""
tests/test_predictor.py — Unit tests for feature engineering in predictor.py.

WHY TEST predictor.py SEPARATELY:
  build_features() is pure math with no I/O. It's the most likely place for
  silent regressions — if someone changes a formula, the API will still return
  200 OK but predictions will be silently wrong. These tests catch that.

NOTE ON RETURN TYPE:
  build_features() returns a Pandas DataFrame (not a list of lists).
  The DataFrame allows sklearn to use named features matching training,
  preventing the "X does not have valid feature names" warning.
  All tests use .iloc[] for row access and column names for value access.

Run with:
    pytest tests/ -v
"""

import math
import pytest
from src.serving.predictor import build_features

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "wind_speed", "wind_direction", "turbulence_intensity", "num_turbines",
    "wind_direction_sin", "wind_direction_cos",
    "wind_speed_squared", "wind_speed_cubed", "wake_adjusted_wind",
]

BASE_RECORD = {
    "wind_speed": 10.0,
    "wind_direction": 90.0,
    "turbulence_intensity": 0.1,
    "num_turbines": 20,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildFeatures:

    def test_returns_correct_number_of_rows(self):
        records = [BASE_RECORD, BASE_RECORD, BASE_RECORD]
        result = build_features(records, FEATURE_NAMES)
        assert len(result) == 3

    def test_returns_correct_number_of_columns(self):
        result = build_features([BASE_RECORD], FEATURE_NAMES)
        assert len(result.columns) == len(FEATURE_NAMES)

    def test_wind_direction_sin_is_correct(self):
        """90 degrees → sin(90°) = 1.0"""
        result = build_features([BASE_RECORD], FEATURE_NAMES)
        assert math.isclose(
            result.iloc[0]["wind_direction_sin"],
            math.sin(math.radians(90.0)),
            rel_tol=1e-6
        )

    def test_wind_direction_cos_is_correct(self):
        """90 degrees → cos(90°) ≈ 0.0"""
        result = build_features([BASE_RECORD], FEATURE_NAMES)
        assert math.isclose(
            result.iloc[0]["wind_direction_cos"],
            math.cos(math.radians(90.0)),
            abs_tol=1e-6
        )

    def test_wind_speed_squared_is_correct(self):
        """10.0 ** 2 = 100.0"""
        result = build_features([BASE_RECORD], FEATURE_NAMES)
        assert result.iloc[0]["wind_speed_squared"] == pytest.approx(100.0)

    def test_wind_speed_cubed_is_correct(self):
        """10.0 ** 3 = 1000.0"""
        result = build_features([BASE_RECORD], FEATURE_NAMES)
        assert result.iloc[0]["wind_speed_cubed"] == pytest.approx(1000.0)

    def test_wake_adjusted_wind_is_correct(self):
        """wake = wind_speed * (1 - turbulence_intensity) = 10.0 * 0.9 = 9.0"""
        result = build_features([BASE_RECORD], FEATURE_NAMES)
        assert result.iloc[0]["wake_adjusted_wind"] == pytest.approx(9.0)

    def test_feature_order_matches_feature_names(self):
        """
        Columns must be in the exact order the model was trained on.
        A silent reordering produces wrong predictions with no error raised.
        """
        result = build_features([BASE_RECORD], FEATURE_NAMES)
        assert list(result.columns) == FEATURE_NAMES
        assert result.iloc[0]["wind_speed"] == BASE_RECORD["wind_speed"]
        assert result.iloc[0]["num_turbines"] == BASE_RECORD["num_turbines"]

    def test_missing_feature_in_names_raises_value_error(self):
        """If feature_names references a column that wasn't engineered, raise clearly."""
        bad_names = FEATURE_NAMES + ["nonexistent_feature"]
        with pytest.raises(ValueError, match="Missing required engineered feature"):
            build_features([BASE_RECORD], bad_names)

    def test_zero_wind_speed_produces_zero_derived_features(self):
        """Edge case: wind_speed=0 should yield 0 for squared, cubed, wake."""
        record = {**BASE_RECORD, "wind_speed": 0.0}
        result = build_features([record], FEATURE_NAMES)
        assert result.iloc[0]["wind_speed_squared"] == 0.0
        assert result.iloc[0]["wind_speed_cubed"]   == 0.0
        assert result.iloc[0]["wake_adjusted_wind"] == 0.0

    def test_north_wind_direction_sin_cos_symmetry(self):
        """0 degrees (North): sin=0, cos=1"""
        record = {**BASE_RECORD, "wind_direction": 0.0}
        result = build_features([record], FEATURE_NAMES)
        assert math.isclose(result.iloc[0]["wind_direction_sin"], 0.0, abs_tol=1e-6)
        assert math.isclose(result.iloc[0]["wind_direction_cos"], 1.0, rel_tol=1e-6)