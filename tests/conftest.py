"""
conftest.py — Shared pytest fixtures for the entire test suite.

KEY PATCHING DECISIONS:
  1. All patches target "src.serving.app.*" because that is WHERE the names
     are looked up at runtime — not where they were originally defined.

  2. We also patch "src.serving.app.build_features" — the function imported
     INTO app.py. Without this patch, the real build_features() runs during
     tests, receives a list of feature-value lists from our mock model, and
     crashes with "list indices must be integers or slices, not str" because
     it tries to do input_dict["wind_speed"] on a plain list.

  3. build_features mock uses side_effect so it returns an array sized to
     match the number of input records — required for batch count assertions.

  4. model.predict mock also uses side_effect for the same reason.
"""

import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Shared constants (imported by test files)
# ---------------------------------------------------------------------------

MOCK_FEATURE_NAMES = [
    "wind_speed", "wind_direction", "turbulence_intensity", "num_turbines",
    "wind_direction_sin", "wind_direction_cos",
    "wind_speed_squared", "wind_speed_cubed", "wake_adjusted_wind",
]

VALID_PAYLOAD = {
    "wind_speed": 12.5,
    "wind_direction": 180.0,
    "turbulence_intensity": 0.08,
    "num_turbines": 50,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_model(output: float = 5000.0) -> MagicMock:
    """
    Mock sklearn model whose predict() returns an array sized to the input.

    WHY side_effect:
      return_value always returns the same fixed array regardless of batch size.
      side_effect receives the actual input X and returns np.full(len(X), output)
      so /predict_batch with 5 records correctly gets 5 predictions back.
    """
    mock_model = MagicMock()
    mock_model.predict.side_effect = lambda X: np.full(len(X), output)
    return mock_model


def make_mock_build_features(output: float = 5000.0):
    """
    Mock for build_features() imported in app.py.

    WHY WE MOCK THIS:
      app.py calls build_features(records, feature_names) which returns a
      list of feature-value lists. Our mock skips all the math and returns
      a correctly-sized dummy list so model.predict() receives valid input.

    The mock returns [[output, output, ...]] * n_records — shape is correct,
    values don't matter because model.predict is also mocked.
    """
    n_features = len(MOCK_FEATURE_NAMES)

    def _mock_build(records, feature_names):
        return [[output] * n_features for _ in records]

    return _mock_build


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """
    Happy-path client. Model returns 5000.0 for every record.
    Boots once per module — fast and mirrors production startup cost.
    """
    mock_model = make_mock_model(output=5000.0)
    mock_build = make_mock_build_features(output=5000.0)

    with patch("src.serving.app.MODEL_PATH") as mock_model_path, \
         patch("src.serving.app.FEATURE_PATH") as mock_feature_path, \
         patch("src.serving.app.METADATA_PATH") as mock_meta_path, \
         patch("src.serving.app.joblib.load", return_value=mock_model), \
         patch("src.serving.app.build_features", side_effect=mock_build), \
         patch("builtins.open", mock_open(read_data=json.dumps(MOCK_FEATURE_NAMES))):

        mock_model_path.exists.return_value   = True
        mock_feature_path.exists.return_value = True
        mock_meta_path.exists.return_value    = False

        from src.serving.app import app
        with TestClient(app) as c:
            yield c


@pytest.fixture(scope="module")
def negative_output_client():
    """
    Isolated client whose model always returns -999.0.
    Used exclusively by TestDomainConstraints.

    WHY SEPARATE FIXTURE (not patch inside the test):
      Patching state mid-session on a shared client causes state leakage.
      A dedicated fixture with its own TestClient is fully isolated.
    """
    mock_model = make_mock_model(output=-999.0)
    mock_build = make_mock_build_features(output=-999.0)

    with patch("src.serving.app.MODEL_PATH") as mock_model_path, \
         patch("src.serving.app.FEATURE_PATH") as mock_feature_path, \
         patch("src.serving.app.METADATA_PATH") as mock_meta_path, \
         patch("src.serving.app.joblib.load", return_value=mock_model), \
         patch("src.serving.app.build_features", side_effect=mock_build), \
         patch("builtins.open", mock_open(read_data=json.dumps(MOCK_FEATURE_NAMES))):

        mock_model_path.exists.return_value   = True
        mock_feature_path.exists.return_value = True
        mock_meta_path.exists.return_value    = False

        from src.serving.app import app
        with TestClient(app) as c:
            yield c