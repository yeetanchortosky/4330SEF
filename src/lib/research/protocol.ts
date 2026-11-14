export const BASE_YAML = `# config/base.yaml — imported by every model. Do not fork.
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
load_col:
  inter: [user_id, item_id, rating]

# Match RecBole-GNN ml-1m: drop ratings below 3.
val_interval:
  rating: "[3,inf)"
unused_col:
  inter: [rating]

epochs: 500
train_batch_size: 4096
eval_batch_size: 4096
learning_rate: 0.002
embedding_size: 64
stopping_step: 10
valid_metric: NDCG@10
metrics: ["Recall", "NDCG", "Hit", "MRR"]
topk: [10, 20]
metric_decimal_place: 4
train_neg_sample_args:
  distribution: uniform
  sample_num: 1
  dynamic: False
eval_args:
  split: { RS: [0.8, 0.1, 0.1] }
  group_by: user
  order: RO
  mode: full
seed: 2020
reproducibility: True
`;

export const LIGHTGCN_YAML = `# config/lightgcn.yaml
# Start here. RecBole-GNN's best ml-1m point, not the library default.
n_layers: 3
reg_weight: 1e-4
`;

export const HYPER_TEST = `# hyper.test — one factor on top of the tuned point
n_layers choice [1,2,3,4]
reg_weight choice [1e-5,1e-4,1e-3]
learning_rate choice [0.005,0.002,0.001]
`;

export const ADAPTIVE_PY = `"""model/lightgcn_adaptive.py

Subclass, do not fork. Logits start at 0 so softmax is uniform
and step 0 is exactly LightGCN.
"""
import torch
from recbole.model.general_recommender.lightgcn import LightGCN


class LightGCNAdaptive(LightGCN):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.layer_logits = torch.nn.Parameter(torch.zeros(self.n_layers + 1))

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        stacked = torch.stack(embeddings_list, dim=1)  # [N, K+1, D]
        alpha = torch.softmax(self.layer_logits, dim=0)
        return (stacked * alpha.view(1, -1, 1)).sum(dim=1)
`;

export const SIDE_PY = `"""model/lightgcn_side.py

Limited fusion: a linear map of a multi-hot genre vector added to E0.
Genres come from item metadata, never from test edges.
You must build genre_table aligned to RecBole's internal item ids
(index 0 is padding). Shape [n_items, n_genres], L1-normalized rows.
"""
import torch
import torch.nn as nn
from recbole.model.general_recommender.lightgcn import LightGCN


class LightGCNSide(LightGCN):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        n_genres = int(config["n_genres"])
        self.genre_proj = nn.Linear(n_genres, self.latent_dim, bias=False)
        # Register a buffer you fill in a setup hook after id mapping.
        self.register_buffer("genre_table", torch.zeros(self.n_items, n_genres))

    def get_ego_embeddings(self):
        ego = super().get_ego_embeddings()
        user, item = torch.split(ego, [self.n_users, self.n_items])
        item = item + self.genre_proj(self.genre_table)
        return torch.cat([user, item], dim=0)
`;

export const RUN_PY = `from recbole.quick_start import run_recbole

# Week-1 gate. Then swap dataset to ml-1m and config_file_list.
run_recbole(
    model="LightGCN",
    dataset="ml-100k",
    config_file_list=["config/base.yaml", "config/lightgcn.yaml"],
)

# Extension: pass the class, not a string, until it is registered.
# from model.lightgcn_adaptive import LightGCNAdaptive
# run_recbole(model=LightGCNAdaptive, dataset="ml-1m", config_file_list=[...])
`;

export const COLD_STEPS = [
  "Train with the shared yaml. Keep the best validation checkpoint. Do not refit per slice.",
  "From the training loader, count interactions per internal user id and item id.",
  "Cold user: count ≤ 5. Cold item: count ≤ 5. Warm and head are the complements.",
  "For each test user, take the full-sort scores, mask training items, cut top 10 and top 20.",
  "Compute Recall, NDCG, and Hit inside each slice. Users with no relevant items in that slice are skipped, and the skip count is reported.",
  "Degradation = slice metric / all-user metric − 1. The headline is this ratio.",
  "Append one row per model × seed × slice to results/runs.csv. Charts read only that file.",
];

export const OWNERS = [
  { name: "Lo Kam Hung", owns: "dataset yaml, ItemKNN, BPR", file: "config/base.yaml" },
  { name: "Yeung Kwan Ho", owns: "NeuMF tuning and both extensions", file: "model/lightgcn_adaptive.py" },
  { name: "Man Timothy Tsz Hin", owns: "LightGCN, NGCF, baseline table", file: "config/lightgcn.yaml" },
  { name: "Tam Ho Chung", owns: "SASRec, cold-start slice, coordination", file: "scripts/cold_start_report.py" },
  { name: "Suen Ching", owns: "runs.csv, charts, literature map", file: "results/runs.csv" },
];
