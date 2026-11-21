"""Train the image <-> KG alignment on the cached NIH features, evaluate it against baselines
and ablations on the held-out test split, save the checkpoint, and write reports/.

Usage: uv run python scripts/train_alignment.py   (after scripts/extract_features.py)
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import UTC, datetime

import numpy as np
import torch

from medvision.align.data import load_feature_sets
from medvision.align.evaluate import (
    case_retrieval,
    classifier_scores,
    embed_images,
    hit_at_k,
    l2_normalise,
    node_scores,
    per_finding_auroc,
    probe_scores,
    random_retrieval,
)
from medvision.align.graph_encoder import build_graph_tensors
from medvision.align.model import finding_node_ids
from medvision.align.train import train_alignment
from medvision.config import REPO_ROOT, get_labels_map, get_settings
from medvision.kg.build import build_graph

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("train_alignment")

SEEDS = (0, 1, 2)
KS = (1, 5, 10)
NODE_KS = (1, 3, 5)


def summarise(model, test, train, cfg) -> dict:
    scores = node_scores(model, test.feats)
    auroc = per_finding_auroc(scores, test.masks, test.finding_ids, cfg.min_positives_for_auroc)
    index_emb = embed_images(model, train.feats)
    query_emb = embed_images(model, test.feats)
    return {
        "auroc": auroc,
        "mean_auroc": float(np.mean(list(auroc.values()))),
        "hit_at_k": hit_at_k(scores, test.masks, NODE_KS),
        "retrieval": case_retrieval(query_emb, test, index_emb, train, KS),
    }


def fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    settings = get_settings()
    cfg = settings.align
    labels_map = get_labels_map()
    kg = build_graph(settings.paths.resolve("kg_ontology_dir"), labels_map)
    finding_ids = finding_node_ids(kg)
    features_path = settings.paths.resolve("processed_dir") / cfg.features_file
    data = load_feature_sets(features_path, finding_ids, labels_map)
    train, val, test = data["train"], data["validation"], data["test"]
    logger.info("images: train %d, validation %d, test %d", len(train), len(val), len(test))

    tensors = build_graph_tensors(kg)
    no_edges = dataclasses.replace(
        tensors,
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, tensors.edge_attr.shape[1])),
    )
    variants = {
        "full (graph + set loss + node loss)": (tensors, cfg),
        "no node-level loss": (tensors, cfg.model_copy(update={"node_loss_weight": 0.0})),
        "no graph edges (text + id only)": (no_edges, cfg),
    }

    results: dict[str, list[dict]] = {name: [] for name in variants}
    deployed = None
    for name, (variant_tensors, variant_cfg) in variants.items():
        for seed in SEEDS:
            run_cfg = variant_cfg.model_copy(update={"seed": seed})
            result = train_alignment(variant_tensors, train, val, run_cfg)
            summary = summarise(result.model, test, train, cfg)
            summary["best_epoch"] = result.best_epoch
            results[name].append(summary)
            logger.info(
                "%-40s seed %d: mean AUROC %.3f, hit@1 %.3f, P@5(abnormal) %.3f (epoch %d)",
                name,
                seed,
                summary["mean_auroc"],
                summary["hit_at_k"][1],
                summary["retrieval"]["abnormal"][5],
                result.best_epoch,
            )
            if name.startswith("full") and seed == SEEDS[0]:
                deployed = result

    assert deployed is not None
    checkpoint = settings.paths.resolve("models_dir") / cfg.checkpoint_file
    deployed.model.save(
        checkpoint,
        extra={
            "trained_at": datetime.now(UTC).isoformat(),
            "n_train": len(train),
            "n_validation": len(val),
            "best_epoch": deployed.best_epoch,
        },
    )
    logger.info("Saved %s", checkpoint)

    # ---- baselines (same frozen features, same test split) ----
    full_seed0 = results[next(iter(variants))][0]
    clf = classifier_scores(test, labels_map)
    probe = probe_scores(train, test)
    clf_auroc = per_finding_auroc(
        np.nan_to_num(clf, nan=0.0), test.masks, test.finding_ids, cfg.min_positives_for_auroc
    )
    probe_auroc = per_finding_auroc(
        probe, test.masks, test.finding_ids, cfg.min_positives_for_auroc
    )
    clf_auroc = {
        k: v for k, v in clf_auroc.items() if not np.isnan(clf[:, test.finding_ids.index(k)]).any()
    }
    shared = [k for k in full_seed0["auroc"] if k in clf_auroc and k in probe_auroc]

    raw_retrieval = case_retrieval(
        l2_normalise(test.feats), test, l2_normalise(train.feats), train, KS
    )
    chance = random_retrieval(test, train, KS, cfg.seed)

    # ---- write reports ----
    names = {n: kg.get_node(n).name for n in test.finding_ids}
    lines = [
        "# Image <-> knowledge-graph alignment: evaluation",
        "",
        f"Generated: {datetime.now(UTC).isoformat(timespec='seconds')}",
        "",
        f"Data: NIH ChestX-ray14 sample (Sohaibsoussi/NIH-Chest-X-ray-dataset-small): "
        f"{len(train)} train / {len(val)} validation / {len(test)} test images. Training uses "
        "train, early stopping uses validation, **every number below is on the test split**, "
        "and the retrieval index for evaluation is the train split only.",
        "",
        "**Caveat that affects every row.** The frozen image encoder is torchxrayvision's "
        "DenseNet-121 `all` checkpoint, which was itself trained on NIH ChestX-ray14. These "
        "features have therefore already seen NIH-style labels, and absolute numbers are "
        "optimistic compared with an encoder that had not. The comparisons between rows "
        "(same features, same split) remain fair; the absolute level does not carry over to "
        "other hospitals.",
        "",
        "## 1. Image -> KG finding node: per-finding AUROC",
        "",
        "Score = cosine similarity to the node (aligned model, seed 0), the pretrained "
        "classifier's own output, or a logistic-regression probe fitted on the same train split. "
        "Findings with fewer than "
        f"{cfg.min_positives_for_auroc} test positives or negatives are skipped.",
        "",
        "| Finding | test positives | Aligned | Classifier head | Linear probe |",
        "|---|---|---|---|---|",
    ]
    for node in full_seed0["auroc"]:
        j = test.finding_ids.index(node)
        lines.append(
            f"| {names[node]} | {int(test.masks[:, j].sum())} | {fmt(full_seed0['auroc'][node])} | "
            f"{fmt(clf_auroc[node]) if node in clf_auroc else 'n/a'} | "
            f"{fmt(probe_auroc[node]) if node in probe_auroc else 'n/a'} |"
        )
    lines += [
        "",
        f"Mean over the {len(shared)} findings all three methods cover: "
        f"aligned **{fmt(np.mean([full_seed0['auroc'][k] for k in shared]))}**, "
        f"classifier head **{fmt(np.mean([clf_auroc[k] for k in shared]))}**, "
        f"linear probe **{fmt(np.mean([probe_auroc[k] for k in shared]))}**.",
        "",
        "## 2. Image -> KG node retrieval (hit@k)",
        "",
        "Share of test images whose top-k retrieved finding nodes contain at least one true "
        "finding (seed 0). Normal images count `No Finding` as their true node.",
        "",
        "| k | hit@k |",
        "|---|---|",
    ]
    lines += [f"| {k} | {fmt(v)} |" for k, v in full_seed0["hit_at_k"].items()]
    lines += [
        "",
        "## 3. Similar-case retrieval (precision@k)",
        "",
        "A retrieved train case is relevant if it shares at least one finding with the query. "
        "'Abnormal' keeps only queries with a finding other than `No Finding`, and relevance "
        "then also ignores `No Finding`, so it is the harder and more meaningful column.",
        "",
        "| Space | P@1 all | P@5 all | P@10 all | P@1 abnormal | P@5 abnormal | P@10 abnormal |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, res in (
        ("Aligned (seed 0)", full_seed0["retrieval"]),
        ("Raw DenseNet features, cosine", raw_retrieval),
        ("Random ranking", chance),
    ):
        row = [res["all"][k] for k in KS] + [res["abnormal"][k] for k in KS]
        lines.append(f"| {label} | " + " | ".join(fmt(v) for v in row) + " |")

    lines += [
        "",
        f"## 4. Ablations (mean ± std over seeds {list(SEEDS)})",
        "",
        "| Variant | mean AUROC | hit@1 | P@5 abnormal |",
        "|---|---|---|---|",
    ]
    for name, runs in results.items():

        def ms(values):
            return f"{np.mean(values):.3f} ± {np.std(values):.3f}"

        lines.append(
            f"| {name} | {ms([r['mean_auroc'] for r in runs])} | "
            f"{ms([r['hit_at_k'][1] for r in runs])} | "
            f"{ms([r['retrieval']['abnormal'][5] for r in runs])} |"
        )
    lines += [
        "",
        "## Reading these results",
        "",
        "This file is generated, so numbers are never edited by hand. The author's "
        "interpretation lives in `reports/alignment_notes.md`.",
        "",
    ]
    out = REPO_ROOT / "reports"
    (out / "alignment_eval.md").write_text("\n".join(lines))
    (out / "alignment_eval.json").write_text(
        json.dumps(
            {
                "full_seed0": full_seed0,
                "classifier_auroc": clf_auroc,
                "probe_auroc": probe_auroc,
                "raw_retrieval": raw_retrieval,
                "random_retrieval": chance,
                "variants": results,
            },
            indent=2,
            default=float,
        )
    )
    logger.info("Wrote reports/alignment_eval.md")


if __name__ == "__main__":
    main()
