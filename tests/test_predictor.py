import numpy as np
import pytest

from p3c_orch.predictor import CalibratedRiskEstimator, CurrentMarginPredictor

from .factories import make_observation


def test_outage_risk_decreases_as_margin_increases() -> None:
    estimator = CalibratedRiskEstimator(
        CurrentMarginPredictor(),
        margin_threshold_db=0.0,
        residual_scale_db={"clear": 4.0},
    )
    low, high = estimator.estimate_many(
        [make_observation(current_margin_db=-5.0), make_observation(current_margin_db=5.0)]
    )
    assert low.outage_probability > high.outage_probability
    assert 0.0 < high.outage_probability < 1.0


def test_non_finite_predictor_output_is_rejected() -> None:
    estimator = CalibratedRiskEstimator(
        CurrentMarginPredictor(),
        margin_threshold_db=0.0,
        residual_scale_db={"clear": 4.0},
    )
    with pytest.raises(ValueError, match="non-finite"):
        estimator.estimate(make_observation(current_margin_db=float("nan")))


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan")])
def test_invalid_residual_scale_is_rejected(scale: float) -> None:
    estimator = CalibratedRiskEstimator(
        CurrentMarginPredictor(),
        margin_threshold_db=0.0,
        residual_scale_db={"clear": scale},
    )
    with pytest.raises(ValueError, match="positive residual scale"):
        estimator.estimate(make_observation())


def test_predictor_output_length_is_checked() -> None:
    class WrongLengthPredictor:
        def predict_many(self, observations):
            return np.asarray([1.0, 2.0])

    estimator = CalibratedRiskEstimator(
        WrongLengthPredictor(),
        margin_threshold_db=0.0,
        residual_scale_db={"clear": 4.0},
    )
    with pytest.raises(ValueError, match="unexpected number"):
        estimator.estimate(make_observation())


def test_non_finite_margin_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="margin threshold"):
        CalibratedRiskEstimator(
            CurrentMarginPredictor(),
            margin_threshold_db=float("nan"),
            residual_scale_db={"clear": 4.0},
        )


def test_uncalibrated_ablation_uses_hard_threshold() -> None:
    estimator = CalibratedRiskEstimator(
        CurrentMarginPredictor(),
        margin_threshold_db=0.0,
        residual_scale_db={"clear": 4.0},
        calibrated=False,
    )
    risks = estimator.estimate_many(
        [make_observation(current_margin_db=-0.1), make_observation(current_margin_db=0.1)]
    )
    assert [item.outage_probability for item in risks] == [1.0, 0.0]
