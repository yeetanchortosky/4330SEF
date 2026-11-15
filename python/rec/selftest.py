"""Same checks as src/lib/rec/selftest.ts."""

from __future__ import annotations

import math
import time

from .data import DATA
from .engine import create_session, default_config, evaluate, train_epoch


def run(label: str, model: str, **extra) -> dict:
    cfg = {**default_config(), "model": model, "epochs": 12, **extra}
    session = create_session(DATA, cfg)
    if model == "knn":
        metrics = evaluate(session)
        print(label, "recall", f"{metrics['all']['recall']:.3f}", "cold", f"{metrics['cold']['recall']:.3f}", "n", metrics["cold"]["n"])
        return metrics
    first = 0.0
    for epoch in range(cfg["epochs"]):
        loss = train_epoch(session)
        if epoch == 0:
            first = loss
        if not math.isfinite(loss):
            raise RuntimeError(f"{label} nonfinite loss {loss}")
    metrics = evaluate(session)
    last = session.loss[-1]
    alphas = ",".join(f"{value:.2f}" for value in metrics["alphas"])
    print(
        label,
        "loss",
        f"{first:.3f}",
        "→",
        f"{last:.3f}",
        "recall",
        f"{metrics['all']['recall']:.3f}",
        "ndcg",
        f"{metrics['all']['ndcg']:.3f}",
        "cold",
        f"{metrics['cold']['recall']:.3f}",
        "warm",
        f"{metrics['warm']['recall']:.3f}",
        "tail",
        f"{metrics['tail']['recall']:.3f}",
        "smooth",
        f"{metrics['smoothness']:.3f}",
        "alpha",
        alphas,
    )
    if not last < first * 0.9:
        raise RuntimeError(f"{label} did not learn")
    if not metrics["all"]["recall"] > 0.04:
        raise RuntimeError(f"{label} recall too low {metrics['all']['recall']}")
    return metrics


def main() -> None:
    started = time.time()
    run("bpr", "bpr")
    run("light", "lightgcn")
    run("side", "side")
    run("ngcf", "ngcf", lr=0.01)
    run("last", "lightgcn", combine="last")
    run("loop", "lightgcn", self_loops=True)
    run("learned", "lightgcn", combine="learned")
    run("knn", "knn")
    cold = sum(1 for role in DATA["roles"] if role == 1)
    print("ms", int((time.time() - started) * 1000))
    print("users", len(DATA["users"]), "items", len(DATA["items"]), "cold", cold)
    print("ok")


if __name__ == "__main__":
    main()
