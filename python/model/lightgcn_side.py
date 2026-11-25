"""python/model/lightgcn_side.py

Limited fusion: a linear map of a multi-hot genre vector added to E0.
Genres come from item metadata, never from test edges.
Aligns genre_table to RecBole's internal item ids (index 0 is padding),
shape [n_items, n_genres], L1-normalized rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from recbole.model.general_recommender.lightgcn import LightGCN


class LightGCNSide(LightGCN):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        matrix_path = Path(config["genre_matrix_path"]) if "genre_matrix_path" in config else Path("results/ml1m_genre_matrix.npz")
        n_genres = int(config["n_genres"]) if "n_genres" in config else 18

        self.genre_proj = nn.Linear(n_genres, self.latent_dim, bias=False)
        self.register_buffer("genre_table", torch.zeros(self.n_items, n_genres, dtype=torch.float32))

        if matrix_path.exists():
            self.load_genre_matrix(matrix_path)

    def load_genre_matrix(self, npz_path: str | Path) -> None:
        with np.load(npz_path) as data:
            key = "genre_matrix" if "genre_matrix" in data.files else data.files[0]
            matrix = data[key].astype(np.float32)

        expected = (self.n_items, self.genre_table.shape[1])
        if matrix.shape != expected:
            raise ValueError(
                f"Genre matrix shape {matrix.shape} does not match expected {expected}. "
                "Ensure config/base.yaml loads interactions only so internal item ids align."
            )
        self.genre_table.copy_(torch.from_numpy(matrix))

    def get_ego_embeddings(self):
        ego = super().get_ego_embeddings()
        user, item = torch.split(ego, [self.n_users, self.n_items])
        item = item + self.genre_proj(self.genre_table)
        return torch.cat([user, item], dim=0)
