import numpy as np

from melting_tank.config import load_config
from melting_tank.evaluation import calculate_metrics, deployment_gate, select_threshold


def test_ng_metrics_and_gate() -> None:
    kpi = load_config().kpi
    y_true = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = calculate_metrics(y_true, probability, 0.5, kpi)
    assert metrics["ng_recall"] == 1.0
    assert metrics["ng_precision"] == 1.0
    normal = calculate_metrics(np.array([0, 0]), np.array([0.1, 0.2]), 0.5, kpi)
    assert deployment_gate(metrics, normal, kpi)["approved"] is True


def test_threshold_uses_highest_f2_among_eligible_candidates() -> None:
    kpi = load_config().kpi
    y_true = np.array([1] * 10 + [0] * 10)
    probability = np.array([0.95] * 9 + [0.55] + [0.70] * 4 + [0.10] * 6)
    selection = select_threshold(y_true, probability, kpi)
    assert selection.validation_kpi_satisfied is True
    assert selection.metrics["ng_recall"] >= kpi.ng_recall_min
    assert selection.metrics["ng_precision"] >= kpi.ng_precision_min


def test_threshold_reports_when_validation_kpi_has_no_candidate() -> None:
    kpi = load_config().kpi
    y_true = np.array([1, 0, 0, 0])
    probability = np.array([0.1, 0.9, 0.8, 0.7])
    selection = select_threshold(y_true, probability, kpi)
    assert selection.validation_kpi_satisfied is False

