from dataclasses import dataclass

import pytest

from medvision.eval import differential as differential_module
from medvision.eval.differential import evaluate_differential_ranking
from medvision.kg.graph import KGEdge, KGNode, KnowledgeGraph


@dataclass
class _FakePrediction:
    node_id: str
    prob: float


class _FakeClassifier:
    """Returns whatever raw-probability dict was stashed against this "image" key."""

    def __init__(self, probs_by_image: dict[str, dict[str, float]]) -> None:
        self.probs_by_image = probs_by_image

    def predict(self, image: str) -> list[_FakePrediction]:
        return [_FakePrediction(node_id=n, prob=p) for n, p in self.probs_by_image[image].items()]


class _IdentityCalibration:
    def apply(self, node_id: str, raw_prob: float) -> float:
        return raw_prob


@dataclass
class _FakeModels:
    classifier: _FakeClassifier
    calibration: _IdentityCalibration
    graph: KnowledgeGraph


def _toy_graph() -> KnowledgeGraph:
    nodes = [
        KGNode(id="F_CONSOLIDATION", type="Finding", name="Consolidation", definition="d"),
        KGNode(id="F_EDEMA", type="Finding", name="Edema", definition="d"),
        KGNode(id="D_PNEUMONIA", type="Disease", name="Pneumonia", definition="d"),
        KGNode(id="D_CHF", type="Disease", name="Heart Failure", definition="d"),
    ]
    edges = [
        KGEdge(src="F_CONSOLIDATION", rel="suggestive_of", dst="D_PNEUMONIA", weight=0.9),
        KGEdge(src="F_EDEMA", rel="suggestive_of", dst="D_CHF", weight=0.9),
    ]
    return KnowledgeGraph(nodes, edges)


@pytest.fixture
def patched_models(monkeypatch):
    def _install(probs_by_image: dict[str, dict[str, float]]):
        fake = _FakeModels(
            classifier=_FakeClassifier(probs_by_image),
            calibration=_IdentityCalibration(),
            graph=_toy_graph(),
        )
        monkeypatch.setattr(differential_module, "get_models", lambda: fake)
        return fake

    return _install


def test_recall_at_1_when_disease_always_top_ranked(patched_models):
    patched_models(
        {
            "pneumonia_case_1": {"F_CONSOLIDATION": 0.9},
            "pneumonia_case_2": {"F_CONSOLIDATION": 0.8},
        }
    )
    images_and_labels = [("pneumonia_case_1", True), ("pneumonia_case_2", True)]
    result = evaluate_differential_ranking(images_and_labels, "D_PNEUMONIA")

    assert result.n_positive == 2
    assert result.recall_at_1 == pytest.approx(1.0)
    assert result.recall_at_3 == pytest.approx(1.0)
    assert result.recall_at_any == pytest.approx(1.0)


def test_recall_at_1_zero_when_disease_never_activated(patched_models):
    patched_models(
        {
            "pneumonia_case_1": {"F_EDEMA": 0.9},  # only activates CHF, not pneumonia
        }
    )
    result = evaluate_differential_ranking([("pneumonia_case_1", True)], "D_PNEUMONIA")
    assert result.recall_at_1 == 0.0
    assert result.recall_at_3 == 0.0
    assert result.recall_at_any == 0.0


def test_recall_at_3_but_not_at_1_when_disease_ranks_lower(patched_models):
    # Both findings activate; CHF should outrank pneumonia since edema's weight+prob is higher
    patched_models(
        {
            "case": {"F_CONSOLIDATION": 0.3, "F_EDEMA": 0.9},
        }
    )
    result = evaluate_differential_ranking([("case", True)], "D_PNEUMONIA")
    assert result.recall_at_1 == 0.0  # CHF is #1, not pneumonia
    assert result.recall_at_3 == pytest.approx(1.0)  # but pneumonia is still in the (short) list


def test_false_alarm_rate_on_negative_cases(patched_models):
    patched_models(
        {
            "normal_1": {"F_CONSOLIDATION": 0.9},  # wrongly fires pneumonia at #1 (false alarm)
            "normal_2": {"F_EDEMA": 0.9},  # correctly doesn't rank pneumonia at #1
        }
    )
    images_and_labels = [("normal_1", False), ("normal_2", False)]
    result = evaluate_differential_ranking(images_and_labels, "D_PNEUMONIA")

    assert result.n_negative == 2
    assert result.false_alarm_at_1 == pytest.approx(0.5)


def test_empty_positive_or_negative_sets_do_not_crash(patched_models):
    patched_models({"case": {"F_CONSOLIDATION": 0.9}})
    result = evaluate_differential_ranking([("case", True)], "D_PNEUMONIA")
    assert result.n_negative == 0
    assert result.false_alarm_at_1 == 0.0
