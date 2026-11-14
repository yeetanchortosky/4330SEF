# ml-1m split statistics (not graded metrics)

Atomic files live in `dataset/ml-1m/`. RecBole’s processed S3 URL returned HTTP 403 on the project machine, so the files were converted from the official GroupLens archive ([ml-1m.zip](https://files.grouplens.org/datasets/movielens/ml-1m.zip)) with `scripts/convert_movielens_1m.py`.

Protocol (from `config/base.yaml`): ratings ≥ 3, random 8:1:1 split, seed 2020, `valid_metric` NDCG@10. Scripts must call `init_seed(2020)` before `dataset.build()`, because `order: RO` shuffles the interactions before they are cut. Counts below are from that seeded split and were confirmed identical on a second run.

| Quantity | Count |
| --- | ---: |
| Raw official ratings / users / movies | 1,000,209 / 6,040 / 3,883 |
| Interactions after ratings ≥ 3 | 836,478 |
| Users after filter (excludes RecBole `[PAD]` id 0) | 6,039 |
| Items after filter (excludes `[PAD]`) | 3,628 |
| Train / valid / test interactions | 674,674 / 80,902 / 80,902 |
| Natural cold users (≤ 5 training edges) | 2 |
| Warm users (> 5 training edges) | 6,037 |
| Natural cold items (≤ 5 training edges) | 372 |
| Warm items (> 5 training edges) | 3,228 |
| Items with zero training edges | 28 |

One official user (and 255 movies) disappear after the rating filter because they have no remaining ratings ≥ 3. RecBole’s `user_num` / `item_num` include a padding id, so they print as 6,040 and 3,629.

Machine-readable copy: `ml1m_split_stats.json`. Slice membership: `ml1m_cold_item_slices.json`.

## Decision (14 Nov 2026 HKT)

The group chose to **headline the ml-1m cold analysis on rarely seen movies**, not new users. After a graded run, `scripts/cold_start_report.py` will score the 372-item slice, and the 28 movies with no training edges as a separate slice, from the same validation checkpoint. The two natural cold users stay in the log only. The 18-user truncated browser cohort is not reused.
