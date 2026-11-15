"""LightGCN, BPR, NGCF-lite, genre fusion, and ItemKNN.

Direct port of src/lib/rec/engine.ts. Arrays stay float32 so a run on the
same seed follows the browser trainer.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .data import DATA

F32 = np.float32
I32 = np.int32


def default_config() -> dict[str, Any]:
    return {
        "model": "lightgcn",
        "layers": 3,
        "combine": "uniform",
        "self_loops": False,
        "dim": 16,
        "reg": 1e-4,
        "lr": 0.02,
        "epochs": 24,
        "seed": 2020,
    }


def _i32(value: int) -> int:
    value &= 0xFFFFFFFF
    if value >= 0x80000000:
        value -= 0x100000000
    return value


def _u32(value: int) -> int:
    return value & 0xFFFFFFFF


def _imul(left: int, right: int) -> int:
    return _i32(_u32(left) * _u32(right))


def mulberry32(seed: int):
    state = _u32(seed)

    def rng() -> float:
        nonlocal state
        signed = _i32(state + 0x6D2B79F5)
        state = _u32(signed)
        mixed = _imul(signed ^ (_u32(signed) >> 15), 1 | signed)
        mixed = _i32(mixed + _imul(mixed ^ (_u32(mixed) >> 7), 61 | mixed)) ^ mixed
        mixed = _i32(mixed)
        return _u32(mixed ^ (_u32(mixed) >> 14)) / 4294967296

    return rng


def _xavier(arr: np.ndarray, fan: int, rng) -> None:
    limit = math.sqrt(6 / (fan + fan))
    for index in range(arr.size):
        arr.ravel()[index] = F32((rng() * 2 - 1) * limit)


def _empty(n: int) -> np.ndarray:
    return np.zeros(n, dtype=F32)


def _softmax(raw: np.ndarray) -> np.ndarray:
    shifted = raw - float(raw.max()) if raw.size else raw
    expo = np.exp(shifted.astype(np.float64)).astype(F32)
    total = float(expo.sum())
    if total == 0:
        return expo
    return (expo / F32(total)).astype(F32)


class Session:
    def __init__(self, pack: dict[str, Any]):
        self.cfg = pack["cfg"]
        self.n_u = pack["n_u"]
        self.n_i = pack["n_i"]
        self.dim = pack["dim"]
        self.layers = pack["layers"]
        self.loss: list[float] = pack["loss"]
        self.recall: list[float] = pack["recall"]
        self.epoch = 0
        self.metrics = None
        self._pack = pack


def build_graph(data: dict, self_loops: bool) -> dict[str, Any]:
    n_u = len(data["users"])
    n_i = len(data["items"])
    user_deg = np.zeros(n_u, dtype=I32)
    item_deg = np.zeros(n_i, dtype=I32)
    for user, item in data["train"]:
        user_deg[user] += 1
        item_deg[item] += 1
    user_self = np.zeros(n_u, dtype=F32)
    item_self = np.zeros(n_i, dtype=F32)
    user_den = np.zeros(n_u, dtype=F32)
    item_den = np.zeros(n_i, dtype=F32)
    for user in range(n_u):
        degree = int(user_deg[user]) + (1 if self_loops else 0)
        user_den[user] = degree
        user_self[user] = (1 / degree) if self_loops and degree > 0 else 0
    for item in range(n_i):
        degree = int(item_deg[item]) + (1 if self_loops else 0)
        item_den[item] = degree
        item_self[item] = (1 / degree) if self_loops and degree > 0 else 0
    user_ptr = np.zeros(n_u + 1, dtype=I32)
    item_ptr = np.zeros(n_i + 1, dtype=I32)
    user_ptr[1:] = np.cumsum(user_deg, dtype=I32)
    item_ptr[1:] = np.cumsum(item_deg, dtype=I32)
    user_items = np.zeros(len(data["train"]), dtype=I32)
    user_w = np.zeros(len(data["train"]), dtype=F32)
    item_users = np.zeros(len(data["train"]), dtype=I32)
    item_w = np.zeros(len(data["train"]), dtype=F32)
    user_fill = user_ptr.copy()
    item_fill = item_ptr.copy()
    for user, item in data["train"]:
        du = float(user_den[user])
        di = float(item_den[item])
        weight = 1 / math.sqrt(du * di) if du > 0 and di > 0 else 0
        pu = int(user_fill[user])
        user_fill[user] += 1
        user_items[pu] = item
        user_w[pu] = weight
        pi = int(item_fill[item])
        item_fill[item] += 1
        item_users[pi] = user
        item_w[pi] = weight
    return {
        "n_u": n_u,
        "n_i": n_i,
        "user_ptr": user_ptr,
        "user_items": user_items,
        "user_w": user_w,
        "item_ptr": item_ptr,
        "item_users": item_users,
        "item_w": item_w,
        "user_self": user_self,
        "item_self": item_self,
    }


def propagate(graph: dict, src_u: np.ndarray, src_i: np.ndarray, dst_u: np.ndarray, dst_i: np.ndarray, dim: int) -> None:
    dst_u.fill(0)
    dst_i.fill(0)
    n_u = graph["n_u"]
    n_i = graph["n_i"]
    for user in range(n_u):
        base = user * dim
        scale = float(graph["user_self"][user])
        if scale:
            dst_u[base : base + dim] += F32(scale) * src_u[base : base + dim]
        for pointer in range(int(graph["user_ptr"][user]), int(graph["user_ptr"][user + 1])):
            item_base = int(graph["user_items"][pointer]) * dim
            weight = F32(graph["user_w"][pointer])
            dst_u[base : base + dim] += weight * src_i[item_base : item_base + dim]
    for item in range(n_i):
        base = item * dim
        scale = float(graph["item_self"][item])
        if scale:
            dst_i[base : base + dim] += F32(scale) * src_i[base : base + dim]
        for pointer in range(int(graph["item_ptr"][item]), int(graph["item_ptr"][item + 1])):
            user_base = int(graph["item_users"][pointer]) * dim
            weight = F32(graph["item_w"][pointer])
            dst_i[base : base + dim] += weight * src_u[user_base : user_base + dim]


def _make_adam(n: int) -> dict[str, Any]:
    return {"m": _empty(n), "v": _empty(n), "t": 0}


def _adam_step(params: np.ndarray, grad: np.ndarray, adam: dict[str, Any], lr: float) -> None:
    adam["t"] += 1
    b1 = 0.9
    b2 = 0.999
    bc1 = 1 - b1 ** adam["t"]
    bc2 = 1 - b2 ** adam["t"]
    moment = adam["m"]
    velocity = adam["v"]
    moment[:] = b1 * moment + (1 - b1) * grad
    velocity[:] = b2 * velocity + (1 - b2) * grad * grad
    params -= F32(lr) * (moment / F32(bc1)) / (np.sqrt(velocity / F32(bc2)) + F32(1e-8))
    grad.fill(0)


def _layer_weights(pack: dict) -> np.ndarray:
    layers = pack["layers"]
    weights = _empty(layers + 1)
    if pack["cfg"]["combine"] == "last":
        weights[layers] = 1
        return weights
    if pack["cfg"]["combine"] == "learned" and pack["alpha"] is not None:
        return _softmax(pack["alpha"])
    uniform = 1 / (layers + 1)
    weights[:] = uniform
    return weights


def _materialize_items(pack: dict) -> None:
    pack["item_ego"][:] = pack["item_emb"]
    genre_w = pack["genre_w"]
    if genre_w is None:
        return
    dim = pack["dim"]
    n_g = pack["n_g"]
    for item, record in enumerate(pack["data"]["items"]):
        genres = record["g"]
        if not genres:
            continue
        scale = 1 / len(genres)
        base = item * dim
        for genre in genres:
            pack["item_ego"][base : base + dim] += F32(scale) * genre_w[np.arange(dim) * n_g + genre]


def _apply_ngcf(ego: np.ndarray, neigh: np.ndarray, weight: np.ndarray, pre: np.ndarray, out: np.ndarray, count: int, dim: int) -> None:
    for index in range(count):
        base = index * dim
        pre[base : base + dim] = ego[base : base + dim] + neigh[base : base + dim]
        for row in range(dim):
            total = float(np.dot(weight[row * dim : (row + 1) * dim], pre[base : base + dim]))
            out[base + row] = total if total >= 0 else 0.2 * total


def _forward(pack: dict) -> dict[str, list]:
    graph = pack["graph"]
    dim = pack["dim"]
    layers = pack["layers"]
    user_states = [pack["user_emb"]]
    item_states = [pack["item_ego"]]
    pre_u: list[np.ndarray] = []
    pre_i: list[np.ndarray] = []
    for layer in range(1, layers + 1):
        next_u = _empty(pack["n_u"] * dim)
        next_i = _empty(pack["n_i"] * dim)
        propagate(graph, user_states[layer - 1], item_states[layer - 1], next_u, next_i, dim)
        if pack["cfg"]["model"] == "ngcf" and pack["ngcf_w"] is not None:
            weight = pack["ngcf_w"][layer - 1]
            out_u = _empty(pack["n_u"] * dim)
            out_i = _empty(pack["n_i"] * dim)
            before_u = _empty(pack["n_u"] * dim)
            before_i = _empty(pack["n_i"] * dim)
            _apply_ngcf(user_states[layer - 1], next_u, weight, before_u, out_u, pack["n_u"], dim)
            _apply_ngcf(item_states[layer - 1], next_i, weight, before_i, out_i, pack["n_i"], dim)
            pre_u.append(before_u)
            pre_i.append(before_i)
            user_states.append(out_u)
            item_states.append(out_i)
        else:
            user_states.append(next_u)
            item_states.append(next_i)
    return {"Eu": user_states, "Ei": item_states, "preU": pre_u, "preI": pre_i}


def _combine(pack: dict, fwd: dict, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    final_u = _empty(pack["n_u"] * pack["dim"])
    final_i = _empty(pack["n_i"] * pack["dim"])
    for layer, alpha in enumerate(weights):
        if not alpha:
            continue
        final_u += F32(alpha) * fwd["Eu"][layer]
        final_i += F32(alpha) * fwd["Ei"][layer]
    return final_u, final_i


def _leaky_backward(out: np.ndarray, grad_out: np.ndarray, pre: np.ndarray, grad_pre: np.ndarray, weight: np.ndarray, grad_w: np.ndarray, dim: int) -> None:
    count = out.size // dim
    for index in range(count):
        base = index * dim
        hidden = np.empty(dim, dtype=F32)
        for row in range(dim):
            hidden[row] = grad_out[base + row] if out[base + row] >= 0 else 0.2 * grad_out[base + row]
        for row in range(dim):
            for col in range(dim):
                grad_pre[base + col] += weight[row * dim + col] * hidden[row]
                grad_w[row * dim + col] += hidden[row] * pre[base + col]


def _ngcf_backward(pack: dict, layer: int, fwd: dict, grad_out_u: np.ndarray, grad_out_i: np.ndarray, grad_prev_u: np.ndarray, grad_prev_i: np.ndarray) -> None:
    weight = pack["ngcf_w"][layer - 1]
    grad_w = pack["g_ngcf"][layer - 1]
    dim = pack["dim"]
    pre_u = fwd["preU"][layer - 1]
    pre_i = fwd["preI"][layer - 1]
    grad_pre_u = _empty(grad_out_u.size)
    grad_pre_i = _empty(grad_out_i.size)
    _leaky_backward(fwd["Eu"][layer], grad_out_u, pre_u, grad_pre_u, weight, grad_w, dim)
    _leaky_backward(fwd["Ei"][layer], grad_out_i, pre_i, grad_pre_i, weight, grad_w, dim)
    neigh_u = grad_pre_u.copy()
    neigh_i = grad_pre_i.copy()
    grad_prev_u += grad_pre_u
    grad_prev_i += grad_pre_i
    back_u = _empty(pack["n_u"] * dim)
    back_i = _empty(pack["n_i"] * dim)
    propagate(pack["graph"], neigh_u, neigh_i, back_u, back_i, dim)
    grad_prev_u += back_u
    grad_prev_i += back_i


def _backprop_layers(pack: dict, fwd: dict, weights: np.ndarray, grad_final_u: np.ndarray, grad_final_i: np.ndarray) -> np.ndarray:
    layers = pack["layers"]
    dim = pack["dim"]
    grad_u = []
    grad_i = []
    for alpha in weights:
        gu = _empty(pack["n_u"] * dim)
        gi = _empty(pack["n_i"] * dim)
        if alpha:
            gu[:] = F32(alpha) * grad_final_u
            gi[:] = F32(alpha) * grad_final_i
        grad_u.append(gu)
        grad_i.append(gi)
    if pack["alpha"] is not None and pack["g_alpha"] is not None:
        dots = np.zeros(layers + 1, dtype=F32)
        for layer in range(layers + 1):
            dots[layer] = float(np.dot(grad_final_u, fwd["Eu"][layer]) + np.dot(grad_final_i, fwd["Ei"][layer]))
        acc = float(np.dot(weights, dots))
        for layer in range(layers + 1):
            pack["g_alpha"][layer] += F32(weights[layer] * (dots[layer] - acc))
    for layer in range(layers, 0, -1):
        if pack["cfg"]["model"] == "ngcf" and pack["ngcf_w"] is not None:
            _ngcf_backward(pack, layer, fwd, grad_u[layer], grad_i[layer], grad_u[layer - 1], grad_i[layer - 1])
        else:
            through_u = _empty(pack["n_u"] * dim)
            through_i = _empty(pack["n_i"] * dim)
            propagate(pack["graph"], grad_u[layer], grad_i[layer], through_u, through_i, dim)
            grad_u[layer - 1] += through_u
            grad_i[layer - 1] += through_i
    pack["g_user"] += grad_u[0]
    return grad_i[0]


def _sample_neg(pack: dict, user: int) -> int:
    n_i = pack["n_i"]
    seen = pack["seen"]
    rng = pack["rng"]
    for _ in range(30):
        item = int(rng() * n_i)
        if not seen[user * n_i + item]:
            return item
    for item in range(n_i):
        if not seen[user * n_i + item]:
            return item
    return 0


def create_session(data: dict | None = None, cfg: dict | None = None) -> Session:
    data = DATA if data is None else data
    cfg = {**default_config(), **(cfg or {})}
    n_u = len(data["users"])
    n_i = len(data["items"])
    n_g = len(data["genreNames"])
    dim = int(cfg["dim"])
    layers = 0 if cfg["model"] in ("bpr", "knn") else int(cfg["layers"])
    rng = mulberry32(int(cfg["seed"]))
    user_emb = _empty(n_u * dim)
    item_emb = _empty(n_i * dim)
    _xavier(user_emb, dim, rng)
    _xavier(item_emb, dim, rng)
    genre_w = _empty(dim * n_g) if cfg["model"] == "side" else None
    if genre_w is not None:
        _xavier(genre_w, n_g, rng)
    learned = cfg["combine"] == "learned" and cfg["model"] not in ("bpr", "knn") and layers > 0
    alpha = _empty(layers + 1) if learned else None
    ngcf_w = None
    if cfg["model"] == "ngcf":
        ngcf_w = []
        for _ in range(layers):
            weight = _empty(dim * dim)
            _xavier(weight, dim, rng)
            ngcf_w.append(weight)
    seen = np.zeros(n_u * n_i, dtype=np.uint8)
    train_by_user: list[list[int]] = [[] for _ in range(n_u)]
    test_by_user: list[list[int]] = [[] for _ in range(n_u)]
    item_deg = np.zeros(n_i, dtype=I32)
    for user, item in data["train"]:
        seen[user * n_i + item] = 1
        train_by_user[user].append(item)
        item_deg[item] += 1
    for user, item in data["test"]:
        test_by_user[user].append(item)
    tail = (item_deg <= 5).astype(np.uint8)
    graph = build_graph(data, False if cfg["model"] == "ngcf" else bool(cfg["self_loops"]))
    train_users = np.array([pair[0] for pair in data["train"]], dtype=I32)
    train_items = np.array([pair[1] for pair in data["train"]], dtype=I32)
    pack = {
        "data": data,
        "graph": graph,
        "seen": seen,
        "train_users": train_users,
        "train_items": train_items,
        "test_by_user": test_by_user,
        "train_by_user": train_by_user,
        "item_deg": item_deg,
        "tail": tail,
        "cfg": cfg,
        "layers": layers,
        "dim": dim,
        "n_u": n_u,
        "n_i": n_i,
        "n_g": n_g,
        "user_emb": user_emb,
        "item_emb": item_emb,
        "g_user": _empty(n_u * dim),
        "g_item": _empty(n_i * dim),
        "a_user": _make_adam(n_u * dim),
        "a_item": _make_adam(n_i * dim),
        "genre_w": genre_w,
        "g_genre": _empty(genre_w.size) if genre_w is not None else None,
        "a_genre": _make_adam(genre_w.size) if genre_w is not None else None,
        "alpha": alpha,
        "g_alpha": _empty(alpha.size) if alpha is not None else None,
        "a_alpha": _make_adam(alpha.size) if alpha is not None else None,
        "ngcf_w": ngcf_w,
        "g_ngcf": [_empty(weight.size) for weight in ngcf_w] if ngcf_w else None,
        "a_ngcf": [_make_adam(weight.size) for weight in ngcf_w] if ngcf_w else None,
        "item_ego": _empty(n_i * dim),
        "rng": rng,
        "loss": [],
        "recall": [],
        "epoch": 0,
        "metrics": None,
    }
    return Session(pack)


def train_epoch(session: Session) -> float:
    pack = session._pack
    if pack["cfg"]["model"] == "knn":
        metrics = _evaluate_pack(pack)
        pack["metrics"] = metrics
        session.metrics = metrics
        session.epoch = 1
        return 0.0
    _materialize_items(pack)
    order = list(range(pack["train_users"].size))
    rng = pack["rng"]
    for index in range(len(order) - 1, 0, -1):
        swap = int(rng() * (index + 1))
        order[index], order[swap] = order[swap], order[index]
    loss_sum = 0.0
    seen_n = 0
    dim = pack["dim"]
    reg = float(pack["cfg"]["reg"])
    batch = 512
    for start in range(0, len(order), batch):
        end = min(len(order), start + batch)
        fwd = _forward(pack)
        weights = _layer_weights(pack)
        final_u, final_i = _combine(pack, fwd, weights)
        grad_u = _empty(final_u.size)
        grad_i = _empty(final_i.size)
        for pointer in range(start, end):
            idx = order[pointer]
            user = int(pack["train_users"][idx])
            positive = int(pack["train_items"][idx])
            negative = _sample_neg(pack, user)
            user_base = user * dim
            pos_base = positive * dim
            neg_base = negative * dim
            score_pos = float(np.dot(final_u[user_base : user_base + dim], final_i[pos_base : pos_base + dim]))
            score_neg = float(np.dot(final_u[user_base : user_base + dim], final_i[neg_base : neg_base + dim]))
            margin = max(-30.0, min(30.0, score_pos - score_neg))
            sigmoid = 1 / (1 + math.exp(-margin))
            diff = sigmoid - 1
            loss_sum += -math.log(max(sigmoid, 1e-8))
            user_vec = final_u[user_base : user_base + dim]
            pos_vec = final_i[pos_base : pos_base + dim]
            neg_vec = final_i[neg_base : neg_base + dim]
            grad_u[user_base : user_base + dim] += F32(diff) * pos_vec - F32(diff) * neg_vec
            grad_i[pos_base : pos_base + dim] += F32(diff) * user_vec
            grad_i[neg_base : neg_base + dim] -= F32(diff) * user_vec
            ego_u = pack["user_emb"][user_base : user_base + dim]
            ego_pos = pack["item_emb"][pos_base : pos_base + dim]
            ego_neg = pack["item_emb"][neg_base : neg_base + dim]
            loss_sum += reg * float(np.dot(ego_u, ego_u) + np.dot(ego_pos, ego_pos) + np.dot(ego_neg, ego_neg))
            pack["g_user"][user_base : user_base + dim] += F32(2 * reg) * ego_u
            pack["g_item"][pos_base : pos_base + dim] += F32(2 * reg) * ego_pos
            pack["g_item"][neg_base : neg_base + dim] += F32(2 * reg) * ego_neg
            seen_n += 1
        item_ego_grad = _backprop_layers(pack, fwd, weights, grad_u, grad_i)
        if pack["genre_w"] is not None and pack["g_genre"] is not None:
            n_g = pack["n_g"]
            for item in range(pack["n_i"]):
                base = item * dim
                pack["g_item"][base : base + dim] += item_ego_grad[base : base + dim]
                genres = pack["data"]["items"][item]["g"]
                if not genres:
                    continue
                scale = 1 / len(genres)
                for genre in genres:
                    pack["g_genre"][np.arange(dim) * n_g + genre] += F32(scale) * item_ego_grad[base : base + dim]
        else:
            pack["g_item"] += item_ego_grad
        lr = float(pack["cfg"]["lr"])
        _adam_step(pack["user_emb"], pack["g_user"], pack["a_user"], lr)
        _adam_step(pack["item_emb"], pack["g_item"], pack["a_item"], lr)
        if pack["genre_w"] is not None:
            _adam_step(pack["genre_w"], pack["g_genre"], pack["a_genre"], lr)
        if pack["alpha"] is not None:
            _adam_step(pack["alpha"], pack["g_alpha"], pack["a_alpha"], lr * 0.5)
        if pack["ngcf_w"] is not None:
            for layer, weight in enumerate(pack["ngcf_w"]):
                _adam_step(weight, pack["g_ngcf"][layer], pack["a_ngcf"][layer], lr)
    mean = loss_sum / max(1, seen_n)
    pack["loss"].append(mean)
    pack["epoch"] += 1
    session.epoch = pack["epoch"]
    return mean


def _empty_slice() -> dict[str, float]:
    return {"recall": 0.0, "ndcg": 0.0, "hr": 0.0, "recall20": 0.0, "ndcg20": 0.0, "hr20": 0.0, "n": 0}


def _add_slice(bucket: dict, at10: dict, at20: dict) -> None:
    bucket["n"] += 1
    bucket["recall"] += at10["recall"]
    bucket["ndcg"] += at10["ndcg"]
    bucket["hr"] += at10["hit"]
    bucket["recall20"] += at20["recall"]
    bucket["ndcg20"] += at20["ndcg"]
    bucket["hr20"] += at20["hit"]


def _finish(bucket: dict) -> dict:
    count = bucket["n"]
    if not count:
        return bucket
    return {key: (value / count if key != "n" else count) for key, value in bucket.items()}


def _topk(scores: np.ndarray, banned: np.ndarray, user: int, n_i: int, k: int) -> np.ndarray:
    order = np.arange(n_i)

    def key(item: int):
        blocked = 1 if banned[user * n_i + item] else 0
        return (blocked, -float(scores[item]))

    ranked = sorted(order.tolist(), key=key)
    return np.asarray(ranked[:k], dtype=I32)


def _rank_metrics(ranked: np.ndarray, relevant: set[int], k: int) -> dict[str, float]:
    hits = 0
    dcg = 0.0
    for rank, item in enumerate(ranked[:k]):
        if int(item) in relevant:
            hits += 1
            dcg += 1 / math.log2(rank + 2)
    ideal = min(k, len(relevant))
    idcg = sum(1 / math.log2(rank + 2) for rank in range(ideal))
    return {
        "recall": hits / len(relevant) if relevant else 0.0,
        "ndcg": dcg / idcg if idcg else 0.0,
        "hit": 1.0 if hits else 0.0,
    }


def _knn_scores(pack: dict) -> np.ndarray:
    n_u = pack["n_u"]
    n_i = pack["n_i"]
    similarity = np.zeros((n_i, n_i), dtype=F32)
    for items in pack["train_by_user"]:
        for left in items:
            for right in items:
                if left != right:
                    similarity[left, right] += 1
    degrees = pack["item_deg"].astype(np.float64)
    denom = np.sqrt(degrees[:, None] * degrees[None, :])
    np.divide(similarity, denom, out=similarity, where=denom > 0)
    scores = np.zeros((n_u, n_i), dtype=F32)
    for user, items in enumerate(pack["train_by_user"]):
        if items:
            scores[user] = similarity[items].sum(axis=0)
    return scores


def _evaluate_pack(pack: dict) -> dict[str, Any]:
    _materialize_items(pack)
    fwd = _forward(pack)
    weights = _layer_weights(pack)
    if pack["cfg"]["model"] == "knn":
        knn = _knn_scores(pack)
        final_u = _empty(0)
        final_i = knn.reshape(-1)
    else:
        final_u, final_i = _combine(pack, fwd, weights)
    n_u = pack["n_u"]
    n_i = pack["n_i"]
    dim = pack["dim"]
    buckets = {name: _empty_slice() for name in ("all", "cold", "warm", "tail", "head")}
    covered = np.zeros(n_i, dtype=np.uint8)
    roles = pack["data"]["roles"]
    examples = []
    want_cold = next((index for index, role in enumerate(roles) if role == 1), -1)
    want_warm = next((index for index, role in enumerate(roles) if role == 0), -1)
    for user in range(n_u):
        tests = pack["test_by_user"][user]
        if not tests:
            continue
        if pack["cfg"]["model"] == "knn":
            scores = final_i[user * n_i : (user + 1) * n_i]
        else:
            user_vec = final_u[user * dim : (user + 1) * dim]
            scores = final_i.reshape(n_i, dim) @ user_vec
        ranked = _topk(scores, pack["seen"], user, n_i, 20)
        covered[ranked[:10]] = 1
        relevant = set(tests)
        at10 = _rank_metrics(ranked, relevant, 10)
        at20 = _rank_metrics(ranked, relevant, 20)
        _add_slice(buckets["all"], at10, at20)
        _add_slice(buckets["cold"] if roles[user] == 1 else buckets["warm"], at10, at20)
        tail_rel = {item for item in tests if pack["tail"][item]}
        head_rel = {item for item in tests if not pack["tail"][item]}
        if tail_rel:
            _add_slice(buckets["tail"], _rank_metrics(ranked, tail_rel, 10), _rank_metrics(ranked, tail_rel, 20))
        if head_rel:
            _add_slice(buckets["head"], _rank_metrics(ranked, head_rel, 10), _rank_metrics(ranked, head_rel, 20))
        if user in (want_cold, want_warm):
            picks = [{"item": int(item), "score": float(scores[int(item)])} for item in ranked[:5]]
            examples.append({"user": user, "cold": roles[user] == 1, "picks": picks})
    smooth = 0.0
    pairs = 0
    if pack["cfg"]["model"] != "knn":
        rng = mulberry32(7)
        for _ in range(40):
            left = int(rng() * n_u)
            right = int(rng() * n_u)
            if left == right:
                continue
            left_vec = final_u[left * dim : (left + 1) * dim].astype(np.float64)
            right_vec = final_u[right * dim : (right + 1) * dim].astype(np.float64)
            left_norm = math.sqrt(float(np.dot(left_vec, left_vec)))
            right_norm = math.sqrt(float(np.dot(right_vec, right_vec)))
            if left_norm < 1e-6 or right_norm < 1e-6:
                continue
            smooth += float(np.dot(left_vec, right_vec)) / (left_norm * right_norm)
            pairs += 1
    return {
        "all": _finish(buckets["all"]),
        "cold": _finish(buckets["cold"]),
        "warm": _finish(buckets["warm"]),
        "tail": _finish(buckets["tail"]),
        "head": _finish(buckets["head"]),
        "smoothness": smooth / pairs if pairs else 0.0,
        "coverage": float(covered.sum()) / n_i,
        "alphas": [float(value) for value in _layer_weights(pack)],
        "examples": examples,
    }


def evaluate(session: Session) -> dict[str, Any]:
    pack = session._pack
    if pack["cfg"]["model"] != "knn":
        _materialize_items(pack)
    metrics = _evaluate_pack(pack)
    pack["metrics"] = metrics
    pack["recall"].append(metrics["all"]["recall"])
    session.metrics = metrics
    session.recall = pack["recall"]
    session.epoch = pack["epoch"]
    return metrics


def recommend_user(session: Session, user: int, k: int = 8) -> list[dict[str, float]]:
    pack = session._pack
    n_i = pack["n_i"]
    dim = pack["dim"]
    if pack["cfg"]["model"] == "knn":
        scores = _knn_scores(pack)[user]
    else:
        _materialize_items(pack)
        fwd = _forward(pack)
        final_u, final_i = _combine(pack, fwd, _layer_weights(pack))
        user_vec = final_u[user * dim : (user + 1) * dim]
        scores = final_i.reshape(n_i, dim) @ user_vec
    ranked = _topk(scores, pack["seen"], user, n_i, k)
    return [{"item": int(item), "score": float(scores[int(item)])} for item in ranked]
