# 4330SEF

Private working copy of Group 5’s COMP 4330SEF project: LightGCN inside RecBole, with a cold-start slice and two small extensions.

The browser app is Orderline.

- **Study** runs the proposal’s six steps on a MovieLens subset.
- **Lab** is the same trainer with free knobs.
- **Slides** is the 5.5-minute talk.
- **Manual** is the operator’s guide, the RecBole subclasses, and the paper-to-code notes.

Branches:
- `main`: Orderline web app, `ml-1m` atomic dataset (`dataset/ml-1m/`), split stats, `[3629, 18]` genre matrix (`results/ml1m_genre_matrix.npz`), and the official seed-2020 LightGCN run (`results/lightgcn_ml1m_run.json`).
- `Python`: Adds the pure-NumPy pilot port under [python/](python/README.md) plus RecBole subclasses (`python/model/`).
- `Notebook-Hydrogen`: Adds the runnable Hydrogen/Jupyter/Colab notebook (`notebooks/4330SEF_LightGCN_ColdStart.ipynb`), `scripts/cold_start_report.py`, wired `LightGCNSide` genre loading, and the reusable RecBole CLI runner.

## Run

### Browser app (Orderline)

```bash
npm install
npm run dev
```

Open `http://127.0.0.1:8080`. `npm run typecheck` and `npm run build` should pass before you treat a change as done.

### Python pilot & RecBole `ml-1m` pipeline

```bash
# Pilot self-test (numpy only)
cd python && python -m rec.selftest

# Graded RecBole run + cold-item slice scoring (from repo root, with recbole + torch)
python python/scripts/run_recbole_example.py --model LightGCN --dataset ml-1m --seed 2020
python scripts/cold_start_report.py --checkpoint saved/LightGCN-Nov-14-2026_21-18-31.pth
```

Or open `notebooks/4330SEF_LightGCN_ColdStart.ipynb` in Hydrogen / Jupyter / Google Colab to run the full pipeline end-to-end.

## Completed `ml-1m` artifacts

- `dataset/ml-1m/`: Official GroupLens `ml-1m` converted to RecBole atomic files (836,478 interactions after rating ≥ 3, 6,039 users, 3,628 items).
- `results/ml1m_cold_item_slices.json`: 372 cold items (≤5 train edges) and 28 zero-edge items.
- `results/ml1m_genre_matrix.npz`: L1-normalized `[3629, 18]` genre matrix aligned to RecBole internal item IDs.
- `results/lightgcn_ml1m_run.json`: Official LightGCN seed-2020 run (Test Recall@10 `0.1857`, NDCG@10 `0.2505`).

## Remaining before 6 Dec 2026 submission

1. Score `saved/LightGCN-Nov-14-2026_21-18-31.pth` with `scripts/cold_start_report.py` into `results/runs.csv`.
2. Run seeds `2021` and `2022` for LightGCN, plus baselines (`BPR`, `ItemKNN`, `NGCF`, `NeuMF`, `SASRec`) and both extensions (`LightGCNAdaptive`, `LightGCNSide`).
3. Run the one-factor ablation sweep (`config/hyper.test`) and generate final report charts from `results/runs.csv`.
4. Submit `SessionP08_Group5 Final Report.pdf`, slides, and code zip by 6 Dec 2026, 23:59.
