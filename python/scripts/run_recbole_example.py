"""Reusable RecBole runner for COMP 4330SEF baselines and LightGCN extensions.

Run from the repository root (or python/ with --config paths adjusted):
  python python/scripts/run_recbole_example.py --model LightGCN --dataset ml-1m --seed 2020
  python python/scripts/run_recbole_example.py --model LightGCNAdaptive --dataset ml-1m --seed 2020
  python python/scripts/run_recbole_example.py --model LightGCNSide --dataset ml-1m --seed 2020
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve_model(name: str):
    if name == "LightGCNAdaptive":
        from model.lightgcn_adaptive import LightGCNAdaptive

        return LightGCNAdaptive
    if name == "LightGCNSide":
        from model.lightgcn_side import LightGCNSide

        return LightGCNSide
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a RecBole experiment under the Group 5 base protocol.")
    parser.add_argument("--model", default="LightGCN", help="Model name or LightGCNAdaptive / LightGCNSide")
    parser.add_argument("--dataset", default="ml-1m", help="RecBole dataset name (e.g. ml-100k, ml-1m)")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override (e.g. 1 for smoke test)")
    parser.add_argument(
        "--config",
        nargs="+",
        default=["config/base.yaml", "config/lightgcn.yaml"],
        help="Config YAML files relative to repo root",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint path to score cold slices")
    args = parser.parse_args()

    from recbole.quick_start import run_recbole

    config_dict: dict = {"seed": args.seed, "reproducibility": True}
    if args.epochs is not None:
        config_dict["epochs"] = args.epochs
    if args.model == "LightGCNSide":
        config_dict["genre_matrix_path"] = str(ROOT / "results" / "ml1m_genre_matrix.npz")
        config_dict["n_genres"] = 18

    started = time.time()
    result = run_recbole(
        model=resolve_model(args.model),
        dataset=args.dataset,
        config_file_list=args.config,
        config_dict=config_dict,
    )
    elapsed = time.time() - started
    print(f"Finished {args.model} on {args.dataset} (seed={args.seed}) in {elapsed:.1f}s")
    print("Result:", result)


if __name__ == "__main__":
    main()
