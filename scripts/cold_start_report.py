"""Score a saved RecBole checkpoint on ml-1m cold-item slices and append to results/runs.csv.

Follows the COMP 4330SEF Group 5 protocol:
  - Uses the exact saved validation checkpoint (never retrains per slice).
  - Evaluates full-sort rankings with training interactions masked.
  - Reports Recall@10, Recall@20, NDCG@10, NDCG@20, Hit@10 across:
      all, cold_item (<=5 train edges, 372 items), zero_item (0 train edges, 28 items), warm_item.
  - Appends one row per slice to results/runs.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

RUNS_COLUMNS = [
    "model",
    "dataset",
    "seed",
    "layers",
    "combine",
    "self_loop",
    "dim",
    "reg",
    "slice",
    "recall10",
    "recall20",
    "ndcg10",
    "ndcg20",
    "hr10",
    "seconds",
    "gpu_mb",
    "epoch",
]


def load_slices(slices_path: Path) -> dict[str, set[int]]:
    with slices_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    cold_ids = {int(entry["internal_id"]) for entry in raw.get("cold_items", [])}
    zero_ids = {int(entry["internal_id"]) for entry in raw.get("zero_train_items", [])}
    return {
        "cold_item": cold_ids,
        "zero_item": zero_ids,
    }


def rank_metrics(ranked: np.ndarray, relevant: set[int], k: int) -> tuple[float, float, float]:
    if not relevant:
        return 0.0, 0.0, 0.0
    hits = 0
    dcg = 0.0
    for rank, item_id in enumerate(ranked[:k]):
        if int(item_id) in relevant:
            hits += 1
            dcg += 1.0 / math.log2(rank + 2)
    ideal = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal))
    recall = hits / len(relevant)
    ndcg = (dcg / idcg) if idcg > 0.0 else 0.0
    hr = 1.0 if hits > 0 else 0.0
    return recall, ndcg, hr


def evaluate_slices(
    checkpoint_path: Path,
    slices_path: Path,
    batch_size: int = 256,
) -> tuple[dict[str, Any], dict[str, dict[str, float | int]]]:
    import torch
    from recbole.quick_start import load_data_and_model
    from recbole.utils import init_seed

    # Ensure deterministic evaluation matching config/base.yaml
    init_seed(2020, True)
    config, model, dataset, train_data, _valid_data, test_data = load_data_and_model(
        model_file=str(checkpoint_path),
    )
    model.eval()
    device = config["device"]

    slices = load_slices(slices_path)
    cold_set = slices["cold_item"]
    zero_set = slices["zero_item"]

    # Build train history and test ground truth per internal user id
    train_inter = train_data.dataset.inter_feat
    train_uids = train_inter[dataset.uid_field].numpy()
    train_iids = train_inter[dataset.iid_field].numpy()
    train_hist: dict[int, list[int]] = {}
    for u, i in zip(train_uids, train_iids):
        train_hist.setdefault(int(u), []).append(int(i))

    test_inter = test_data.dataset.inter_feat
    test_uids = test_inter[dataset.uid_field].numpy()
    test_iids = test_inter[dataset.iid_field].numpy()
    test_gt: dict[int, set[int]] = {}
    for u, i in zip(test_uids, test_iids):
        test_gt.setdefault(int(u), set()).add(int(i))

    user_ids = sorted(test_gt.keys())
    accum = {
        name: {"recall10": 0.0, "recall20": 0.0, "ndcg10": 0.0, "ndcg20": 0.0, "hr10": 0.0, "n": 0, "skipped": 0}
        for name in ("all", "cold_item", "zero_item", "warm_item")
    }

    with torch.no_grad():
        for start in range(0, len(user_ids), batch_size):
            batch_users = user_ids[start : start + batch_size]
            interaction = {dataset.uid_field: torch.tensor(batch_users, dtype=torch.long, device=device)}
            scores = model.full_sort_predict(interaction)
            scores = scores.view(len(batch_users), -1).cpu().numpy()

            for row_idx, uid in enumerate(batch_users):
                row_scores = scores[row_idx]
                row_scores[0] = -np.inf  # mask RecBole [PAD] item 0
                seen = train_hist.get(uid, [])
                if seen:
                    row_scores[seen] = -np.inf

                top20 = np.argpartition(row_scores, -20)[-20:]
                top20 = top20[np.argsort(row_scores[top20])[::-1]]

                gt_all = test_gt[uid]
                gt_cold = gt_all & cold_set
                gt_zero = gt_all & zero_set
                gt_warm = gt_all - cold_set - zero_set

                for slice_name, gt_slice in (
                    ("all", gt_all),
                    ("cold_item", gt_cold),
                    ("zero_item", gt_zero),
                    ("warm_item", gt_warm),
                ):
                    if not gt_slice:
                        accum[slice_name]["skipped"] += 1
                        continue
                    r10, n10, h10 = rank_metrics(top20, gt_slice, 10)
                    r20, n20, _ = rank_metrics(top20, gt_slice, 20)
                    bucket = accum[slice_name]
                    bucket["n"] += 1
                    bucket["recall10"] += r10
                    bucket["recall20"] += r20
                    bucket["ndcg10"] += n10
                    bucket["ndcg20"] += n20
                    bucket["hr10"] += h10

    summary: dict[str, dict[str, float | int]] = {}
    for slice_name, bucket in accum.items():
        n = int(bucket["n"])
        summary[slice_name] = {
            "n_users": n,
            "skipped_users": int(bucket["skipped"]),
            "recall10": round(bucket["recall10"] / n, 4) if n else 0.0,
            "recall20": round(bucket["recall20"] / n, 4) if n else 0.0,
            "ndcg10": round(bucket["ndcg10"] / n, 4) if n else 0.0,
            "ndcg20": round(bucket["ndcg20"] / n, 4) if n else 0.0,
            "hr10": round(bucket["hr10"] / n, 4) if n else 0.0,
        }

    meta = {
        "model": str(config["model"]),
        "dataset": str(config["dataset"]),
        "seed": int(config["seed"]),
        "layers": int(config["n_layers"]) if "n_layers" in config else 0,
        "dim": int(config["embedding_size"]) if "embedding_size" in config else 64,
        "reg": float(config["reg_weight"]) if "reg_weight" in config else 0.0,
    }
    return meta, summary


def append_runs_csv(
    csv_path: Path,
    meta: dict[str, Any],
    summary: dict[str, dict[str, float | int]],
    combine: str,
    self_loop: bool,
    seconds: float,
    gpu_mb: float,
    epoch: int,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNS_COLUMNS)
        if write_header:
            writer.writeheader()
        for slice_name, metrics in summary.items():
            writer.writerow(
                {
                    "model": meta["model"],
                    "dataset": meta["dataset"],
                    "seed": meta["seed"],
                    "layers": meta["layers"],
                    "combine": combine,
                    "self_loop": str(self_loop).lower(),
                    "dim": meta["dim"],
                    "reg": meta["reg"],
                    "slice": slice_name,
                    "recall10": metrics["recall10"],
                    "recall20": metrics["recall20"],
                    "ndcg10": metrics["ndcg10"],
                    "ndcg20": metrics["ndcg20"],
                    "hr10": metrics["hr10"],
                    "seconds": round(seconds, 2),
                    "gpu_mb": round(gpu_mb, 1),
                    "epoch": epoch,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a RecBole checkpoint on cold-item slices.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to saved RecBole .pth checkpoint")
    parser.add_argument(
        "--slices",
        type=Path,
        default=Path("results/ml1m_cold_item_slices.json"),
        help="Path to cold-item slice JSON",
    )
    parser.add_argument("--out", type=Path, default=Path("results/runs.csv"), help="Output CSV path")
    parser.add_argument("--combine", default="uniform", choices=["uniform", "last", "learned", "none"])
    parser.add_argument("--self-loop", action="store_true")
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--gpu-mb", type=float, default=0.0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    meta, summary = evaluate_slices(args.checkpoint, args.slices, batch_size=args.batch_size)
    append_runs_csv(
        args.out,
        meta,
        summary,
        combine=args.combine,
        self_loop=args.self_loop,
        seconds=args.seconds,
        gpu_mb=args.gpu_mb,
        epoch=args.epoch,
    )
    print(json.dumps({"meta": meta, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
