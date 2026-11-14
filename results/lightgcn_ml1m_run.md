# LightGCN on MovieLens 1M (official run)

This is the graded LightGCN run on the proposal split, not a smoke test or a CPU probe.

- Machine: DESKTOP-951EVM3, NVIDIA GeForce GTX 1660 (6 GB)
- Started 14 Nov 2026, 21:18:08 Hong Kong time; finished 22:43:17 (5,103.9 seconds, about 1 hour 25 minutes)
- Protocol: ratings ≥ 3, random 8:1:1 split, seed 2020, full-sort evaluation, `valid_metric` NDCG@10, early stopping after 10 flat epochs
- Best validation at epoch 125; training stopped after epoch 136
- Checkpoint (not in this commit; the training PC was offline at push time): `D:\4330SEF\saved\LightGCN-Nov-14-2026_21-18-31.pth`

Machine-readable copy: `lightgcn_ml1m_run.json`.

## Best validation

| Metric | Score |
| --- | ---: |
| NDCG@10 | 0.2023 |
| NDCG@20 | 0.2208 |
| Recall@10 | 0.1643 |
| Recall@20 | 0.2537 |
| Hit@10 | 0.7143 |
| Hit@20 | 0.8228 |
| MRR@10 | 0.3613 |
| MRR@20 | 0.369 |

## Test (one evaluation from the best validation checkpoint)

| Metric | Score |
| --- | ---: |
| NDCG@10 | 0.2505 |
| NDCG@20 | 0.2615 |
| Recall@10 | 0.1857 |
| Recall@20 | 0.2777 |
| Hit@10 | 0.7294 |
| Hit@20 | 0.8319 |
| MRR@10 | 0.432 |
| MRR@20 | 0.4394 |

Cold-item scoring of the 372 rarely-seen movies (and the 28 with no training edges) is not in this file yet.
