# 4330SEF

Private working copy of Group 5’s COMP 4330SEF project: LightGCN inside RecBole, with a cold-start slice and two small extensions.

The browser app is Orderline.

- **Study** runs the proposal’s six steps on a MovieLens subset.
- **Lab** is the same trainer with free knobs.
- **Slides** is the 5.5-minute talk.
- **Manual** is the operator’s guide, the RecBole subclasses, and the paper-to-code notes.

The `Python` branch keeps this app and adds a numpy port of the trainer under [python/](python/README.md). `python -m rec.selftest` is the check that the port runs. It is the same 66-user pilot, not the graded ml-1m table.

## Run

```bash
npm install
npm run dev
```

Open `http://127.0.0.1:8080`. `npm run typecheck` and `npm run build` should pass before you treat a change as done.

## What this repository already contains

- A LightGCN trainer that matches RecBole’s update: normalized bipartite propagation, uniform / last / learned layer mix, optional self-loops, BPR, L2 on the ego embeddings.
- ItemKNN, BPR, an NGCF-lite, and genre fusion on the same subset.
- Recall, NDCG, and Hit at 10 and 20, plus a cold / warm / tail / head cut.
- RecBole yaml and the two subclasses, in the Manual tab and under `src/lib/research/protocol.ts`.
- The proposal and the course brief, under `attachments/`.

The subset is MovieLens latest-small, ratings 4 and 5, 66 users. Eighteen of them were truncated to 3 training edges so the cold slice is not empty. Those numbers are a pilot. They are not the graded table.

## Still missing before the graded run

RecBole downloads `ml-1m` by itself the first time you set `dataset: ml-1m` (atomic files: `ml-1m.inter`, `ml-1m.item`, `ml-1m.user`). You do not commit that zip. Confirm the second dataset name on the RecBole dataset list before you lock it; the proposal allows Amazon-Beauty or Gowalla, and only Beauty has categories for the genre arm.

1. A Python environment with RecBole, and RecBole-GNN if you import its LightGCN. A GPU machine or a Colab session that can finish 500-epoch runs. The brief warns that Colab limits are real.
2. The ml-100k smoke run, then ml-1m with ratings ≥ 3, split 8:1:1, full sort, three to five seeds. `results/runs.csv` does not exist yet.
3. A count of users and items that naturally have ≤ 5 training edges in that split. Do not reuse the truncated cohort from this browser subset.
4. NeuMF and SASRec, trained in RecBole, not in this browser core.
5. The genre matrix aligned to RecBole’s internal item ids (index 0 is padding), built from `ml-1m.item`. Same job for Beauty categories if that is the second set. Gowalla has no categories, so it cannot test fusion.
6. Training time and GPU memory, the proposal’s secondary metrics.
7. The validation rule in practice: pick the checkpoint on validation NDCG@10, touch test once, and call a gap a tie when it sits inside the seed spread.
8. A PowerPoint or PDF of the six talk slides for OLE. The in-app deck is not a file the submission form will accept. Initial slides were due 15 Nov 2026; the talk is Week 12/13, P08, Thursday 17:00, C0411. The report PDF must be named `SessionP08_Group5 Final Report.pdf`, due 6 Dec 2026, one zip, late work is not accepted.
9. A group check that P08 was not reassigned. If the tutorial changes, the report filename changes with it.

## Do not write into the report

- Subset Recall as if it were ml-1m.
- A learned-mix result and a genre-fusion result as one model. The first table runs them separately.
- A cold threshold other than five, unless you froze the new number before looking at test metrics.
