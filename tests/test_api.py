"""
tests/test_api.py — Integration tests for the Wind Power Prediction API.

Run with:
    pytest tests/ -v

WHAT THESE TESTS COVER:
  - /health endpoint liveness and model-loaded status
  - /predict happy path: status, type safety, response contract, middleware
  - /predict domain constraint: negative output MUST be clipped to exactly 0.0
  - /predict validation: all Pydantic field constraints enforced
  - /predict_batch: count parity, non-negative outputs, edge cases
"""

import pytest

# VALID_PAYLOAD lives in conftest.py — pytest makes it available as a module-level
# import because conftest.py is on the Python path during test collection.
from conftest import VALID_PAYLOAD


# =============================================================================
# /health
# =============================================================================

class TestHealthEndpoint:

    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_confirms_model_is_loaded(self, client):
        assert client.get("/health").json()["model_loaded"] is True

    def test_includes_model_version_field(self, client):
        assert "model_version" in client.get("/health").json()


# =============================================================================
# /predict — Happy Path
# =============================================================================

class TestPredictHappyPath:

    def test_valid_request_returns_200(self, client):
        assert client.post("/predict", json=VALID_PAYLOAD).status_code == 200

    def test_prediction_is_a_float(self, client):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert isinstance(body["prediction_kw"], float)

    def test_prediction_is_non_negative(self, client):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert body["prediction_kw"] >= 0.0

    def test_response_includes_model_version(self, client):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "model_version" in body

    def test_response_includes_request_id(self, client):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "request_id" in body
        assert len(body["request_id"]) > 0

    def test_x_request_id_header_is_present(self, client):
        """Middleware must attach X-Request-ID to every response."""
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert "x-request-id" in response.headers

    def test_x_request_id_is_valid_uuid4(self, client):
        """Header value must be a well-formed UUID4 (36 chars including hyphens)."""
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert len(response.headers["x-request-id"]) == 36

    def test_response_body_request_id_matches_header(self, client):
        """
        The request_id in the JSON body must match X-Request-ID in the header.
        If they diverge, tracing is broken — two different IDs for the same request.
        """
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.json()["request_id"] == response.headers["x-request-id"]


# =============================================================================
# /predict — Domain Constraint (Critical Regression Guard)
# =============================================================================

class TestDomainConstraints:
    """
    Uses the dedicated `negative_output_client` fixture whose mock model
    returns -999.0. This verifies max(0.0, ...) without touching the
    shared happy-path client.
    """

    def test_negative_model_output_is_clipped_to_exactly_zero(self, negative_output_client):
        """
        CRITICAL REGRESSION GUARD.
        If someone deletes the max(0.0, ...) line, this test fails and
        blocks the deployment. Asserts exact 0.0 — not just >= 0.
        """
        body = negative_output_client.post("/predict", json=VALID_PAYLOAD).json()
        assert body["prediction_kw"] == 0.0

    def test_batch_negative_outputs_all_clipped_to_zero(self, negative_output_client):
        """Domain constraint must hold for every item in a batch."""
        batch = [VALID_PAYLOAD] * 3
        body = negative_output_client.post("/predict_batch", json=batch).json()
        assert all(p == 0.0 for p in body["predictions_kw"])


# =============================================================================
# /predict — Pydantic Validation Errors
# =============================================================================

class TestPredictValidation:

    def test_negative_wind_speed_returns_422(self, client):
        """Field constraint: ge=0.0"""
        response = client.post("/predict", json={**VALID_PAYLOAD, "wind_speed": -10.0})
        assert response.status_code == 422

    def test_wind_direction_above_360_returns_422(self, client):
        """Field constraint: le=360.0"""
        response = client.post("/predict", json={**VALID_PAYLOAD, "wind_direction": 361.0})
        assert response.status_code == 422

    def test_turbulence_above_1_returns_422(self, client):
        """Field constraint: le=1.0"""
        response = client.post("/predict", json={**VALID_PAYLOAD, "turbulence_intensity": 1.5})
        assert response.status_code == 422

    def test_zero_turbines_returns_422(self, client):
        """Field constraint: gt=0"""
        response = client.post("/predict", json={**VALID_PAYLOAD, "num_turbines": 0})
        assert response.status_code == 422

    def test_missing_required_field_returns_422(self, client):
        response = client.post("/predict", json={"wind_speed": 10.0})
        assert response.status_code == 422

    def test_wrong_type_returns_422(self, client):
        """Sending a string for a float field must be rejected."""
        response = client.post("/predict", json={**VALID_PAYLOAD, "wind_speed": "fast"})
        assert response.status_code == 422


# =============================================================================
# /predict_batch
# =============================================================================

class TestBatchEndpoint:

    def test_valid_batch_returns_200(self, client):
        assert client.post("/predict_batch", json=[VALID_PAYLOAD, VALID_PAYLOAD]).status_code == 200

    def test_response_count_matches_input_size(self, client):
        """
        CRITICAL: count in response must equal number of records sent.
        Catches the bug where mock returns fixed-size array regardless of input.
        """
        batch = [VALID_PAYLOAD] * 5
        body = client.post("/predict_batch", json=batch).json()
        assert body["count"] == 5
        assert len(body["predictions_kw"]) == 5

    def test_single_record_batch_is_valid(self, client):
        """Batch of 1 is a valid edge case."""
        body = client.post("/predict_batch", json=[VALID_PAYLOAD]).json()
        assert body["count"] == 1

    def test_all_batch_predictions_are_non_negative(self, client):
        batch = [VALID_PAYLOAD] * 3
        body = client.post("/predict_batch", json=batch).json()
        assert all(p >= 0.0 for p in body["predictions_kw"])

    def test_empty_batch_returns_400(self, client):
        assert client.post("/predict_batch", json=[]).status_code == 400

    def test_oversized_batch_returns_400(self, client):
        oversized = [VALID_PAYLOAD] * 501
        assert client.post("/predict_batch", json=oversized).status_code == 400

    def test_batch_response_includes_x_request_id_header(self, client):
        response = client.post("/predict_batch", json=[VALID_PAYLOAD])
        assert "x-request-id" in response.headers

    def test_batch_invalid_record_returns_422(self, client):
        """A single invalid record in the batch must reject the entire request."""
        bad_record = {**VALID_PAYLOAD, "wind_speed": -5.0}
        response = client.post("/predict_batch", json=[VALID_PAYLOAD, bad_record])
        assert response.status_code == 422