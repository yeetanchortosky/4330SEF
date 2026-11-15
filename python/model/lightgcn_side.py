"""model/lightgcn_side.py

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
