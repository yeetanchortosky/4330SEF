"""model/lightgcn_adaptive.py

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
