# Python trainer

Numpy port of `src/lib/rec/engine.ts`. `main` is still the Orderline web app. This branch adds the same algorithm in Python.

What runs here, with only numpy:

- BPR, LightGCN (uniform / last / learned layer mix, optional self-loops), genre fusion, NGCF-lite, ItemKNN
- Same 66-user MovieLens latest-small pilot as the browser (ratings ≥ 4, seed 2020). Eighteen users are truncated to 3 training edges. That cohort is not the graded ml-1m run.

What does not run here:

- `model/lightgcn_adaptive.py` and `model/lightgcn_side.py` are the RecBole subclasses from the protocol. They import `recbole` and `torch`. They were syntax-checked only. No ml-1m scores are claimed for them.
- `scripts/run_recbole_example.py` is the RecBole entry. Run it on the machine that already has the venv, from this directory, after `pip install recbole torch`.

## Run the check

```bash
cd python
pip install -r requirements.txt
python -m rec.selftest
```

The check trains 12 epochs. It requires a finite loss, a last loss below 90% of the first (ItemKNN is exempt), and Recall@10 above 0.04. `mulberry32(2020)` matches the JavaScript sampler.

One passing run (numpy 2.2.6, about 16 s):

| model | loss | Recall@10 |
| --- | --- | --- |
| bpr | 0.707 → 0.376 | 0.098 |
| lightgcn | 0.686 → 0.400 | 0.228 |
| side | 0.683 → 0.405 | 0.220 |
| ngcf (lr 0.01) | 0.676 → 0.420 | 0.170 |
| lightgcn last | 0.693 → 0.592 | 0.168 |
| lightgcn self-loops | 0.685 → 0.386 | 0.229 |
| lightgcn learned | 0.686 → 0.351 | 0.231 |
| itemknn | — | 0.192 |

These are pilot numbers. Do not put them in the graded table.
