import numpy as np
import pytest
import torch

from medvision.align.data import NO_FINDING_NODE, FeatureSet
from medvision.align.evaluate import (
    case_retrieval,
    embed_images,
    hit_at_k,
    l2_normalise,
    node_scores,
    per_finding_auroc,
)
from medvision.align.graph_encoder import build_graph_tensors
from medvision.align.loss import multi_positive_nce, same_label_set, symmetric_set_loss
from medvision.align.model import AlignmentModel, finding_node_ids
from medvision.align.train import train_alignment
from medvision.config import AlignConfig, get_labels_map, get_settings
from medvision.kg.build import build_graph


@pytest.fixture(scope="module")
def kg():
    settings = get_settings()
    return build_graph(settings.paths.resolve("kg_ontology_dir"), get_labels_map())


@pytest.fixture(scope="module")
def tensors(kg):
    return build_graph_tensors(kg)


def small_cfg(**overrides) -> AlignConfig:
    base = dict(embed_dim=16, hidden_dim=32, gnn_heads=2, feature_dim=24, epochs=40, patience=40)
    base.update(overrides)
    return AlignConfig(**base)


def synthetic(kg, n: int, seed: int, noise: float = 0.5) -> FeatureSet:
    """Features = sum of one random prototype per true finding, plus noise."""
    rng = np.random.default_rng(seed)
    finding_ids = finding_node_ids(kg)
    protos = np.random.default_rng(123).normal(size=(len(finding_ids), 24)).astype(np.float32)
    masks = np.zeros((n, len(finding_ids)), dtype=np.float32)
    for i in range(n):
        masks[i, rng.choice(len(finding_ids), size=rng.integers(1, 3), replace=False)] = 1
    feats = masks @ protos + noise * rng.normal(size=(n, 24)).astype(np.float32)
    return FeatureSet(
        feats=feats.astype(np.float32),
        masks=masks,
        finding_ids=finding_ids,
        case_ids=[f"c{i}" for i in range(n)],
        label_names=[[] for _ in range(n)],
        xrv_outputs=np.zeros((n, 1), dtype=np.float32),
        xrv_pathologies=["x"],
    )


def test_duplicate_label_sets_are_positives_not_negatives():
    masks = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    positives = same_label_set(masks)
    assert positives.tolist() == [[True, True, False], [True, True, False], [False, False, True]]

    logits = torch.tensor([[5.0, 5.0, 0.0], [5.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    with_duplicates = multi_positive_nce(logits, positives)
    only_diagonal = multi_positive_nce(logits, torch.eye(3, dtype=torch.bool))
    # Rows 0 and 1 are equally close to each other and to themselves; counting the twin as a
    # positive must make that cheaper than counting it as a negative.
    assert with_duplicates < only_diagonal
    expected_row0 = -(
        torch.log(torch.tensor(2 * np.exp(5.0) / (2 * np.exp(5.0) + 1.0), dtype=torch.float32))
    )
    row0 = multi_positive_nce(logits[:1], positives[:1])
    assert torch.isclose(row0, expected_row0, atol=1e-5)


def test_symmetric_set_loss_is_lower_for_aligned_embeddings():
    masks = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    good = torch.nn.functional.normalize(torch.tensor([[1.0, 0], [1, 0], [0, 1], [0, 1]]), dim=-1)
    bad = good[[2, 3, 0, 1]]
    scale = torch.tensor(10.0)
    assert symmetric_set_loss(good, good, masks, scale) < symmetric_set_loss(
        good, bad, masks, scale
    )


def test_graph_tensors_cover_every_node_and_both_edge_directions(kg, tensors):
    assert len(tensors.node_ids) == len(kg)
    assert tensors.edge_index.shape == (2, 2 * len(kg.edges))
    assert tensors.edge_attr.shape[0] == tensors.edge_index.shape[1]
    weights = tensors.edge_attr[:, -1]
    assert weights.min() >= 0 and weights.max() <= 1
    forward, reverse = tensors.edge_attr[0], tensors.edge_attr[1]
    num_rel = (tensors.edge_attr.shape[1] - 1) // 2
    assert forward[:num_rel].sum() == 1 and reverse[num_rel:-1].sum() == 1
    assert tensors.text.shape[0] == len(kg) and (tensors.text.sum(1) > 0).all()


def test_model_embeddings_are_unit_length_and_save_load_roundtrips(kg, tensors, tmp_path):
    model = AlignmentModel(tensors, small_cfg()).eval()
    model.set_findings(finding_node_ids(kg))
    feats = torch.randn(5, 24)
    img = model.image_embeddings(feats)
    nodes = model.node_embeddings()
    assert nodes.shape == (len(kg), 16)
    assert torch.allclose(img.norm(dim=-1), torch.ones(5), atol=1e-5)
    assert torch.allclose(nodes.norm(dim=-1), torch.ones(len(kg)), atol=1e-5)

    path = tmp_path / "m.pt"
    model.save(path)
    loaded = AlignmentModel.load(path)
    assert loaded.finding_ids == model.finding_ids
    assert torch.allclose(loaded.image_embeddings(feats), img, atol=1e-6)
    assert torch.allclose(loaded.node_embeddings(), nodes, atol=1e-6)


def test_graph_edges_influence_node_embeddings(kg, tensors):
    """Zeroing every edge weight/relation must change the embeddings, i.e. the graph is used."""
    model = AlignmentModel(tensors, small_cfg()).eval()
    base = model.node_embeddings()
    model.graph.edge_attr.zero_()
    assert not torch.allclose(model.node_embeddings(), base, atol=1e-4)


def test_training_lowers_loss_and_learns_the_task(kg, tensors):
    train, val, test = synthetic(kg, 600, 1), synthetic(kg, 150, 2), synthetic(kg, 200, 3)
    result = train_alignment(tensors, train, val, small_cfg(lr=3e-3))
    first, last = result.history[0]["train_loss"], result.history[-1]["train_loss"]
    assert last < 0.6 * first

    scores = node_scores(result.model, test.feats)
    hits = hit_at_k(scores, test.masks, (1, 3))
    assert hits[1] > 0.85 and hits[3] >= hits[1]
    aurocs = per_finding_auroc(scores, test.masks, test.finding_ids, min_positives=10)
    assert aurocs and np.mean(list(aurocs.values())) > 0.9


def test_aligned_retrieval_beats_chance(kg, tensors):
    train, val, test = synthetic(kg, 600, 1), synthetic(kg, 150, 2), synthetic(kg, 150, 3)
    result = train_alignment(tensors, train, val, small_cfg(lr=3e-3))
    index_emb = embed_images(result.model, train.feats)
    query_emb = embed_images(result.model, test.feats)
    precision = case_retrieval(query_emb, test, index_emb, train, (5,))
    chance = float((test.masks @ train.masks.T > 0).mean())
    assert precision["all"][5] > chance + 0.3


def test_hit_at_k_and_auroc_helpers():
    scores = np.array([[0.9, 0.1, 0.0], [0.1, 0.2, 0.9]])
    masks = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    assert hit_at_k(scores, masks, (1, 2)) == {1: 0.5, 2: 1.0}

    rng = np.random.default_rng(0)
    y = (rng.random((200, 2)) > 0.5).astype(np.float32)
    s = y + 0.1 * rng.normal(size=y.shape)
    out = per_finding_auroc(s, y, ["a", "b"], min_positives=5)
    assert set(out) == {"a", "b"} and all(v > 0.95 for v in out.values())
    rare = np.zeros((200, 1), dtype=np.float32)
    rare[:3] = 1
    assert per_finding_auroc(rng.random((200, 1)), rare, ["r"], min_positives=5) == {}


def test_l2_normalise_and_no_finding_constant():
    assert np.allclose(np.linalg.norm(l2_normalise(np.random.rand(4, 7)), axis=1), 1.0)
    assert NO_FINDING_NODE == "F_NO_FINDING"
