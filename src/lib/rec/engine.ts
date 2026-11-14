import { MOVIELENS } from "./movielens";

export type RecData = {
  source: string;
  genreNames: readonly string[];
  roles: readonly number[];
  users: readonly string[];
  items: readonly { readonly t: string; readonly g: readonly number[] }[];
  train: readonly (readonly [number, number])[];
  test: readonly (readonly [number, number])[];
};

export const DATA = MOVIELENS as unknown as RecData;

export type ModelName = "bpr" | "lightgcn" | "side" | "ngcf" | "knn";
export type Combine = "uniform" | "last" | "learned";

export type TrainConfig = {
  model: ModelName;
  layers: number;
  combine: Combine;
  selfLoops: boolean;
  dim: number;
  reg: number;
  lr: number;
  epochs: number;
  seed: number;
};

export type Slice = {
  recall: number;
  ndcg: number;
  hr: number;
  recall20: number;
  ndcg20: number;
  hr20: number;
  n: number;
};

export type Metrics = {
  all: Slice;
  cold: Slice;
  warm: Slice;
  tail: Slice;
  head: Slice;
  smoothness: number;
  coverage: number;
  alphas: number[];
  examples: { user: number; cold: boolean; picks: { item: number; score: number }[] }[];
};

export type Session = {
  cfg: TrainConfig;
  nU: number;
  nI: number;
  dim: number;
  layers: number;
  loss: number[];
  recall: number[];
  epoch: number;
  metrics: Metrics | null;
};

type Rng = () => number;

function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function xavier(arr: Float32Array, fan: number, rng: Rng) {
  const lim = Math.sqrt(6 / (fan + fan));
  for (let i = 0; i < arr.length; i++) arr[i] = (rng() * 2 - 1) * lim;
}

type Graph = {
  nU: number;
  nI: number;
  uPtr: Int32Array;
  uItems: Int32Array;
  uW: Float32Array;
  iPtr: Int32Array;
  iUsers: Int32Array;
  iW: Float32Array;
  uSelf: Float32Array;
  iSelf: Float32Array;
};

function buildGraph(data: RecData, selfLoops: boolean): Graph {
  const nU = data.users.length;
  const nI = data.items.length;
  const udeg = new Int32Array(nU);
  const ideg = new Int32Array(nI);
  for (const [u, i] of data.train) {
    udeg[u]++;
    ideg[i]++;
  }
  const uSelf = new Float32Array(nU);
  const iSelf = new Float32Array(nI);
  const uDen = new Float32Array(nU);
  const iDen = new Float32Array(nI);
  for (let u = 0; u < nU; u++) {
    const d = selfLoops ? udeg[u] + 1 : udeg[u];
    uDen[u] = d;
    uSelf[u] = selfLoops && d > 0 ? 1 / d : 0;
  }
  for (let i = 0; i < nI; i++) {
    const d = selfLoops ? ideg[i] + 1 : ideg[i];
    iDen[i] = d;
    iSelf[i] = selfLoops && d > 0 ? 1 / d : 0;
  }
  const uPtr = new Int32Array(nU + 1);
  const iPtr = new Int32Array(nI + 1);
  for (let u = 0; u < nU; u++) uPtr[u + 1] = uPtr[u] + udeg[u];
  for (let i = 0; i < nI; i++) iPtr[i + 1] = iPtr[i] + ideg[i];
  const uItems = new Int32Array(data.train.length);
  const uW = new Float32Array(data.train.length);
  const iUsers = new Int32Array(data.train.length);
  const iW = new Float32Array(data.train.length);
  const uFill = uPtr.slice();
  const iFill = iPtr.slice();
  for (const [u, i] of data.train) {
    const du = uDen[u];
    const di = iDen[i];
    const w = du > 0 && di > 0 ? 1 / Math.sqrt(du * di) : 0;
    const pu = uFill[u]++;
    uItems[pu] = i;
    uW[pu] = w;
    const pi = iFill[i]++;
    iUsers[pi] = u;
    iW[pi] = w;
  }
  return { nU, nI, uPtr, uItems, uW, iPtr, iUsers, iW, uSelf, iSelf };
}

function propagate(graph: Graph, srcU: Float32Array, srcI: Float32Array, dstU: Float32Array, dstI: Float32Array, dim: number) {
  dstU.fill(0);
  dstI.fill(0);
  const { nU, nI, uPtr, uItems, uW, iPtr, iUsers, iW, uSelf, iSelf } = graph;
  for (let u = 0; u < nU; u++) {
    const base = u * dim;
    const s = uSelf[u];
    if (s) {
      for (let d = 0; d < dim; d++) dstU[base + d] += s * srcU[base + d];
    }
    for (let p = uPtr[u]; p < uPtr[u + 1]; p++) {
      const ib = uItems[p] * dim;
      const w = uW[p];
      for (let d = 0; d < dim; d++) dstU[base + d] += w * srcI[ib + d];
    }
  }
  for (let i = 0; i < nI; i++) {
    const base = i * dim;
    const s = iSelf[i];
    if (s) {
      for (let d = 0; d < dim; d++) dstI[base + d] += s * srcI[base + d];
    }
    for (let p = iPtr[i]; p < iPtr[i + 1]; p++) {
      const ub = iUsers[p] * dim;
      const w = iW[p];
      for (let d = 0; d < dim; d++) dstI[base + d] += w * srcU[ub + d];
    }
  }
}

type Adam = { m: Float32Array; v: Float32Array; t: number };

function makeAdam(n: number): Adam {
  return { m: new Float32Array(n), v: new Float32Array(n), t: 0 };
}

function adamStep(p: Float32Array, g: Float32Array, adam: Adam, lr: number) {
  adam.t++;
  const b1 = 0.9;
  const b2 = 0.999;
  const bc1 = 1 - Math.pow(b1, adam.t);
  const bc2 = 1 - Math.pow(b2, adam.t);
  const m = adam.m;
  const v = adam.v;
  for (let i = 0; i < p.length; i++) {
    const gi = g[i];
    m[i] = b1 * m[i] + (1 - b1) * gi;
    v[i] = b2 * v[i] + (1 - b2) * gi * gi;
    p[i] -= (lr * (m[i] / bc1)) / (Math.sqrt(v[i] / bc2) + 1e-8);
  }
  g.fill(0);
}

type Packed = {
  data: RecData;
  graph: Graph;
  seen: Uint8Array;
  trainUsers: Int32Array;
  trainItems: Int32Array;
  testByUser: number[][];
  trainByUser: number[][];
  itemDeg: Int32Array;
  tail: Uint8Array;
  cfg: TrainConfig;
  layers: number;
  dim: number;
  nU: number;
  nI: number;
  nG: number;
  userEmb: Float32Array;
  itemEmb: Float32Array;
  gUser: Float32Array;
  gItem: Float32Array;
  aUser: Adam;
  aItem: Adam;
  genreW: Float32Array | null;
  gGenre: Float32Array | null;
  aGenre: Adam | null;
  alpha: Float32Array | null;
  gAlpha: Float32Array | null;
  aAlpha: Adam | null;
  ngcfW: Float32Array[] | null;
  gNgcf: Float32Array[] | null;
  aNgcf: Adam[] | null;
  itemEgo: Float32Array;
  rng: Rng;
  loss: number[];
  recall: number[];
  epoch: number;
  metrics: Metrics | null;
};

function softmax(raw: Float32Array): Float32Array {
  const out = new Float32Array(raw.length);
  let m = -Infinity;
  for (let i = 0; i < raw.length; i++) m = Math.max(m, raw[i]);
  let s = 0;
  for (let i = 0; i < raw.length; i++) {
    out[i] = Math.exp(raw[i] - m);
    s += out[i];
  }
  for (let i = 0; i < raw.length; i++) out[i] /= s;
  return out;
}

function layerWeights(pack: Packed): Float32Array {
  const k = pack.layers;
  const a = new Float32Array(k + 1);
  if (pack.cfg.combine === "last") {
    a[k] = 1;
    return a;
  }
  if (pack.cfg.combine === "learned" && pack.alpha) return softmax(pack.alpha);
  const u = 1 / (k + 1);
  for (let i = 0; i <= k; i++) a[i] = u;
  return a;
}

function materializeItems(pack: Packed) {
  const { dim, nI, nG, itemEmb, itemEgo, genreW, data } = pack;
  itemEgo.set(itemEmb);
  if (!genreW) return;
  for (let i = 0; i < nI; i++) {
    const gs = data.items[i].g;
    if (!gs.length) continue;
    const scale = 1 / gs.length;
    const base = i * dim;
    for (const g of gs) {
      for (let d = 0; d < dim; d++) itemEgo[base + d] += scale * genreW[d * nG + g];
    }
  }
}

function empty(n: number) {
  return new Float32Array(n);
}

type Fwd = { Eu: Float32Array[]; Ei: Float32Array[]; preU: Float32Array[]; preI: Float32Array[] };

function forward(pack: Packed): Fwd {
  const { graph, dim, layers, nU, nI, userEmb, itemEgo } = pack;
  const Eu: Float32Array[] = [userEmb];
  const Ei: Float32Array[] = [itemEgo];
  const preU: Float32Array[] = [];
  const preI: Float32Array[] = [];
  for (let k = 1; k <= layers; k++) {
    const nu = empty(nU * dim);
    const ni = empty(nI * dim);
    propagate(graph, Eu[k - 1], Ei[k - 1], nu, ni, dim);
    if (pack.cfg.model === "ngcf" && pack.ngcfW) {
      const W = pack.ngcfW[k - 1];
      const ou = empty(nU * dim);
      const oi = empty(nI * dim);
      const pu = empty(nU * dim);
      const pi = empty(nI * dim);
      applyNgcf(Eu[k - 1], nu, W, pu, ou, nU, dim);
      applyNgcf(Ei[k - 1], ni, W, pi, oi, nI, dim);
      preU.push(pu);
      preI.push(pi);
      Eu.push(ou);
      Ei.push(oi);
    } else {
      Eu.push(nu);
      Ei.push(ni);
    }
  }
  return { Eu, Ei, preU, preI };
}

function applyNgcf(ego: Float32Array, neigh: Float32Array, W: Float32Array, pre: Float32Array, out: Float32Array, n: number, dim: number) {
  for (let i = 0; i < n; i++) {
    const b = i * dim;
    for (let d = 0; d < dim; d++) pre[b + d] = ego[b + d] + neigh[b + d];
    for (let r = 0; r < dim; r++) {
      let s = 0;
      const wr = r * dim;
      for (let c = 0; c < dim; c++) s += W[wr + c] * pre[b + c];
      out[b + r] = s >= 0 ? s : 0.2 * s;
    }
  }
}

function combine(pack: Packed, fwd: Fwd, weights: Float32Array) {
  const { nU, nI, dim, layers } = pack;
  const fu = empty(nU * dim);
  const fi = empty(nI * dim);
  for (let k = 0; k <= layers; k++) {
    const a = weights[k];
    if (!a) continue;
    const eu = fwd.Eu[k];
    const ei = fwd.Ei[k];
    for (let i = 0; i < fu.length; i++) fu[i] += a * eu[i];
    for (let i = 0; i < fi.length; i++) fi[i] += a * ei[i];
  }
  return { fu, fi };
}

function dot(a: Float32Array, ao: number, b: Float32Array, bo: number, dim: number) {
  let s = 0;
  for (let d = 0; d < dim; d++) s += a[ao + d] * b[bo + d];
  return s;
}

function axpy(dst: Float32Array, off: number, src: Float32Array, soff: number, scale: number, dim: number) {
  for (let d = 0; d < dim; d++) dst[off + d] += scale * src[soff + d];
}

export function createSession(data: RecData, cfg: TrainConfig): Session {
  const nU = data.users.length;
  const nI = data.items.length;
  const nG = data.genreNames.length;
  const dim = cfg.dim;
  const layers = cfg.model === "bpr" || cfg.model === "knn" ? 0 : cfg.layers;
  const rng = mulberry32(cfg.seed);
  const userEmb = empty(nU * dim);
  const itemEmb = empty(nI * dim);
  xavier(userEmb, dim, rng);
  xavier(itemEmb, dim, rng);
  const useSide = cfg.model === "side";
  const genreW = useSide ? empty(dim * nG) : null;
  if (genreW) xavier(genreW, nG, rng);
  const learned = cfg.combine === "learned" && cfg.model !== "bpr" && cfg.model !== "knn" && layers > 0;
  const alpha = learned ? new Float32Array(layers + 1) : null;
  const ngcf = cfg.model === "ngcf";
  const ngcfW = ngcf
    ? Array.from({ length: layers }, () => {
        const W = empty(dim * dim);
        xavier(W, dim, rng);
        return W;
      })
    : null;
  const seen = new Uint8Array(nU * nI);
  const trainByUser: number[][] = Array.from({ length: nU }, () => []);
  const testByUser: number[][] = Array.from({ length: nU }, () => []);
  const itemDeg = new Int32Array(nI);
  for (const [u, i] of data.train) {
    seen[u * nI + i] = 1;
    trainByUser[u].push(i);
    itemDeg[i]++;
  }
  for (const [u, i] of data.test) testByUser[u].push(i);
  const tail = new Uint8Array(nI);
  for (let i = 0; i < nI; i++) if (itemDeg[i] <= 5) tail[i] = 1;
  const graph = buildGraph(data, cfg.model === "ngcf" ? false : cfg.selfLoops);
  const pack: Packed = {
    data,
    graph,
    seen,
    trainUsers: new Int32Array(data.train.length),
    trainItems: new Int32Array(data.train.length),
    testByUser,
    trainByUser,
    itemDeg,
    tail,
    cfg,
    layers,
    dim,
    nU,
    nI,
    nG,
    userEmb,
    itemEmb,
    gUser: empty(nU * dim),
    gItem: empty(nI * dim),
    aUser: makeAdam(nU * dim),
    aItem: makeAdam(nI * dim),
    genreW,
    gGenre: genreW ? empty(genreW.length) : null,
    aGenre: genreW ? makeAdam(genreW.length) : null,
    alpha,
    gAlpha: alpha ? empty(alpha.length) : null,
    aAlpha: alpha ? makeAdam(alpha.length) : null,
    ngcfW,
    gNgcf: ngcfW ? ngcfW.map((W) => empty(W.length)) : null,
    aNgcf: ngcfW ? ngcfW.map((W) => makeAdam(W.length)) : null,
    itemEgo: empty(nI * dim),
    rng,
    loss: [],
    recall: [],
    epoch: 0,
    metrics: null,
  };
  data.train.forEach(([u, i], idx) => {
    pack.trainUsers[idx] = u;
    pack.trainItems[idx] = i;
  });
  return publicSession(pack);
}

const sessions = new WeakMap<Session, Packed>();

function publicSession(pack: Packed): Session {
  const session: Session = {
    cfg: pack.cfg,
    nU: pack.nU,
    nI: pack.nI,
    dim: pack.dim,
    layers: pack.layers,
    loss: pack.loss,
    recall: pack.recall,
    epoch: 0,
    metrics: null,
  };
  sessions.set(session, pack);
  return session;
}

function sampleNeg(pack: Packed, u: number): number {
  const { nI, seen, rng } = pack;
  for (let t = 0; t < 30; t++) {
    const i = (rng() * nI) | 0;
    if (!seen[u * nI + i]) return i;
  }
  for (let i = 0; i < nI; i++) if (!seen[u * nI + i]) return i;
  return 0;
}

function backpropLayers(pack: Packed, fwd: Fwd, weights: Float32Array, gFu: Float32Array, gFi: Float32Array) {
  const { layers, dim, nU, nI, graph } = pack;
  const Gu: Float32Array[] = [];
  const Gi: Float32Array[] = [];
  for (let k = 0; k < weights.length; k++) {
    const a = weights[k];
    const gu = empty(nU * dim);
    const gi = empty(nI * dim);
    if (a) {
      for (let i = 0; i < gu.length; i++) gu[i] = a * gFu[i];
      for (let i = 0; i < gi.length; i++) gi[i] = a * gFi[i];
    }
    Gu.push(gu);
    Gi.push(gi);
  }
  if (pack.alpha && pack.gAlpha) {
    const dots = new Float32Array(layers + 1);
    for (let k = 0; k <= layers; k++) {
      let s = 0;
      const eu = fwd.Eu[k];
      const ei = fwd.Ei[k];
      for (let i = 0; i < gFu.length; i++) s += gFu[i] * eu[i];
      for (let i = 0; i < gFi.length; i++) s += gFi[i] * ei[i];
      dots[k] = s;
    }
    let acc = 0;
    for (let k = 0; k <= layers; k++) acc += weights[k] * dots[k];
    for (let k = 0; k <= layers; k++) pack.gAlpha[k] += weights[k] * (dots[k] - acc);
  }
  for (let k = layers; k >= 1; k--) {
    if (pack.cfg.model === "ngcf" && pack.ngcfW && pack.gNgcf) {
      ngcfBackward(pack, k, fwd, Gu[k], Gi[k], Gu[k - 1], Gi[k - 1]);
    } else {
      const tu = empty(nU * dim);
      const ti = empty(nI * dim);
      propagate(graph, Gu[k], Gi[k], tu, ti, dim);
      for (let i = 0; i < tu.length; i++) Gu[k - 1][i] += tu[i];
      for (let i = 0; i < ti.length; i++) Gi[k - 1][i] += ti[i];
    }
  }
  for (let i = 0; i < pack.gUser.length; i++) pack.gUser[i] += Gu[0][i];
  return Gi[0];
}

function ngcfBackward(pack: Packed, k: number, fwd: Fwd, gOu: Float32Array, gOi: Float32Array, gPrevU: Float32Array, gPrevI: Float32Array) {
  const W = pack.ngcfW![k - 1];
  const gW = pack.gNgcf![k - 1];
  const dim = pack.dim;
  const pu = fwd.preU[k - 1];
  const pi = fwd.preI[k - 1];
  const ou = fwd.Eu[k];
  const oi = fwd.Ei[k];
  const gPreU = empty(gOu.length);
  const gPreI = empty(gOi.length);
  leakyBackward(ou, gOu, pu, gPreU, W, gW, dim);
  leakyBackward(oi, gOi, pi, gPreI, W, gW, dim);
  const nU = pack.nU;
  const nI = pack.nI;
  const neighGU = empty(nU * dim);
  const neighGI = empty(nI * dim);
  for (let i = 0; i < gPreU.length; i++) {
    gPrevU[i] += gPreU[i];
    neighGU[i] = gPreU[i];
  }
  for (let i = 0; i < gPreI.length; i++) {
    gPrevI[i] += gPreI[i];
    neighGI[i] = gPreI[i];
  }
  const bu = empty(nU * dim);
  const bi = empty(nI * dim);
  propagate(pack.graph, neighGU, neighGI, bu, bi, dim);
  for (let i = 0; i < bu.length; i++) gPrevU[i] += bu[i];
  for (let i = 0; i < bi.length; i++) gPrevI[i] += bi[i];
}

function leakyBackward(out: Float32Array, gOut: Float32Array, pre: Float32Array, gPre: Float32Array, W: Float32Array, gW: Float32Array, dim: number) {
  const n = out.length / dim;
  for (let i = 0; i < n; i++) {
    const b = i * dim;
    const gh = empty(dim);
    for (let r = 0; r < dim; r++) gh[r] = out[b + r] >= 0 || out[b + r] === 0 ? gOut[b + r] : 0.2 * gOut[b + r];
    for (let r = 0; r < dim; r++) {
      for (let c = 0; c < dim; c++) {
        gPre[b + c] += W[r * dim + c] * gh[r];
        gW[r * dim + c] += gh[r] * pre[b + c];
      }
    }
  }
}

export function trainEpoch(session: Session): number {
  const pack = sessions.get(session);
  if (!pack) throw new Error("Session is closed.");
  if (pack.cfg.model === "knn") {
    pack.metrics = evaluatePack(pack);
    session.metrics = pack.metrics;
    session.epoch = 1;
    return 0;
  }
  materializeItems(pack);
  const order = Array.from({ length: pack.trainUsers.length }, (_, i) => i);
  const rng = pack.rng;
  for (let i = order.length - 1; i > 0; i--) {
    const j = (rng() * (i + 1)) | 0;
    const tmp = order[i];
    order[i] = order[j];
    order[j] = tmp;
  }
  const batch = 512;
  let lossSum = 0;
  let seenN = 0;
  const dim = pack.dim;
  const reg = pack.cfg.reg;
  for (let start = 0; start < order.length; start += batch) {
    const end = Math.min(order.length, start + batch);
    const fwd = forward(pack);
    const weights = layerWeights(pack);
    const { fu, fi } = combine(pack, fwd, weights);
    const gFu = empty(fu.length);
    const gFi = empty(fi.length);
    for (let p = start; p < end; p++) {
      const idx = order[p];
      const u = pack.trainUsers[idx];
      const pos = pack.trainItems[idx];
      const neg = sampleNeg(pack, u);
      const ub = u * dim;
      const pb = pos * dim;
      const nb = neg * dim;
      const su = dot(fu, ub, fi, pb, dim);
      const sn = dot(fu, ub, fi, nb, dim);
      const x = Math.max(-30, Math.min(30, su - sn));
      const sig = 1 / (1 + Math.exp(-x));
      const diff = sig - 1;
      lossSum += -Math.log(Math.max(sig, 1e-8));
      axpy(gFu, ub, fi, pb, diff, dim);
      axpy(gFu, ub, fi, nb, -diff, dim);
      axpy(gFi, pb, fu, ub, diff, dim);
      axpy(gFi, nb, fu, ub, -diff, dim);
      const eu = pack.userEmb;
      const ei = pack.itemEmb;
      let regTerm = 0;
      for (let d = 0; d < dim; d++) {
        regTerm += eu[ub + d] * eu[ub + d] + ei[pb + d] * ei[pb + d] + ei[nb + d] * ei[nb + d];
        pack.gUser[ub + d] += 2 * reg * eu[ub + d];
        pack.gItem[pb + d] += 2 * reg * ei[pb + d];
        pack.gItem[nb + d] += 2 * reg * ei[nb + d];
      }
      lossSum += reg * regTerm;
      seenN++;
    }
    const gItemEgo = backpropLayers(pack, fwd, weights, gFu, gFi);
    if (pack.genreW && pack.gGenre) {
      const nG = pack.nG;
      for (let i = 0; i < pack.nI; i++) {
        const b = i * dim;
        for (let d = 0; d < dim; d++) pack.gItem[b + d] += gItemEgo[b + d];
        const gs = pack.data.items[i].g;
        if (!gs.length) continue;
        const scale = 1 / gs.length;
        for (const g of gs) {
          for (let d = 0; d < dim; d++) pack.gGenre[d * nG + g] += scale * gItemEgo[b + d];
        }
      }
    } else {
      for (let i = 0; i < pack.gItem.length; i++) pack.gItem[i] += gItemEgo[i];
    }
    const lr = pack.cfg.lr;
    adamStep(pack.userEmb, pack.gUser, pack.aUser, lr);
    adamStep(pack.itemEmb, pack.gItem, pack.aItem, lr);
    if (pack.genreW && pack.gGenre && pack.aGenre) adamStep(pack.genreW, pack.gGenre, pack.aGenre, lr);
    if (pack.alpha && pack.gAlpha && pack.aAlpha) adamStep(pack.alpha, pack.gAlpha, pack.aAlpha, lr * 0.5);
    if (pack.ngcfW && pack.gNgcf && pack.aNgcf) {
      for (let k = 0; k < pack.ngcfW.length; k++) adamStep(pack.ngcfW[k], pack.gNgcf[k], pack.aNgcf[k], lr);
    }
  }
  const mean = lossSum / Math.max(1, seenN);
  pack.loss.push(mean);
  pack.epoch++;
  session.epoch = pack.epoch;
  session.loss = pack.loss;
  return mean;
}

function slice(): Slice {
  return { recall: 0, ndcg: 0, hr: 0, recall20: 0, ndcg20: 0, hr20: 0, n: 0 };
}

function addSlice(
  s: Slice,
  a: { recall: number; ndcg: number; hit: number },
  b: { recall: number; ndcg: number; hit: number },
) {
  s.n++;
  s.recall += a.recall;
  s.ndcg += a.ndcg;
  s.hr += a.hit;
  s.recall20 += b.recall;
  s.ndcg20 += b.ndcg;
  s.hr20 += b.hit;
}

function finish(s: Slice): Slice {
  if (!s.n) return s;
  const n = s.n;
  return {
    recall: s.recall / n,
    ndcg: s.ndcg / n,
    hr: s.hr / n,
    recall20: s.recall20 / n,
    ndcg20: s.ndcg20 / n,
    hr20: s.hr20 / n,
    n,
  };
}

function topK(scores: Float32Array, banned: Uint8Array, u: number, nI: number, k: number) {
  const idx = new Int32Array(nI);
  for (let i = 0; i < nI; i++) idx[i] = i;
  idx.sort((a, b) => {
    const ba = banned[u * nI + a] ? 1 : 0;
    const bb = banned[u * nI + b] ? 1 : 0;
    if (ba !== bb) return ba - bb;
    return scores[b] - scores[a];
  });
  return idx.subarray(0, k);
}

function rankMetrics(ranked: Int32Array, relevant: Set<number>, k: number) {
  let hits = 0;
  let dcg = 0;
  for (let r = 0; r < ranked.length && r < k; r++) {
    if (relevant.has(ranked[r])) {
      hits++;
      dcg += 1 / Math.log2(r + 2);
    }
  }
  const ideal = Math.min(k, relevant.size);
  let idcg = 0;
  for (let r = 0; r < ideal; r++) idcg += 1 / Math.log2(r + 2);
  return {
    recall: relevant.size ? hits / relevant.size : 0,
    ndcg: idcg ? dcg / idcg : 0,
    hit: hits > 0 ? 1 : 0,
  };
}

function evaluatePack(pack: Packed): Metrics {
  materializeItems(pack);
  const fwd = forward(pack);
  const weights = layerWeights(pack);
  let fu: Float32Array;
  let fi: Float32Array;
  if (pack.cfg.model === "knn") {
    const knn = knnScores(pack);
    fu = knn.fu;
    fi = knn.fi;
  } else {
    const combined = combine(pack, fwd, weights);
    fu = combined.fu;
    fi = combined.fi;
  }
  const { nU, nI, dim, data, seen, testByUser, tail } = pack;
  const all = slice();
  const cold = slice();
  const warm = slice();
  const tailS = slice();
  const headS = slice();
  const k = 20;
  const examples: Metrics["examples"] = [];
  const covered = new Uint8Array(nI);
  const wantCold = data.roles.findIndex((r) => r === 1);
  const wantWarm = data.roles.findIndex((r) => r === 0);
  for (let u = 0; u < nU; u++) {
    const tests = testByUser[u];
    if (!tests.length) continue;
    const scores = empty(nI);
    if (pack.cfg.model === "knn") {
      scores.set(fi.subarray(u * nI, u * nI + nI));
    } else {
      const ub = u * dim;
      for (let i = 0; i < nI; i++) scores[i] = dot(fu, ub, fi, i * dim, dim);
    }
    const ranked = topK(scores, seen, u, nI, k);
    for (let r = 0; r < 10 && r < ranked.length; r++) covered[ranked[r]] = 1;
    const rel = new Set(tests);
    const m10 = rankMetrics(ranked, rel, 10);
    const m20 = rankMetrics(ranked, rel, 20);
    addSlice(all, m10, m20);
    const bucket = data.roles[u] === 1 ? cold : warm;
    addSlice(bucket, m10, m20);
    const tailRel = new Set(tests.filter((i) => tail[i]));
    const headRel = new Set(tests.filter((i) => !tail[i]));
    if (tailRel.size) addSlice(tailS, rankMetrics(ranked, tailRel, 10), rankMetrics(ranked, tailRel, 20));
    if (headRel.size) addSlice(headS, rankMetrics(ranked, headRel, 10), rankMetrics(ranked, headRel, 20));
    if (u === wantCold || u === wantWarm) {
      const picks: { item: number; score: number }[] = [];
      for (let r = 0; r < ranked.length && picks.length < 5; r++) {
        picks.push({ item: ranked[r], score: scores[ranked[r]] });
      }
      examples.push({ user: u, cold: data.roles[u] === 1, picks });
    }
  }
  let coverN = 0;
  for (let i = 0; i < nI; i++) coverN += covered[i];
  let smooth = 0;
  let pairs = 0;
  if (pack.cfg.model !== "knn") {
    const rng = mulberry32(7);
    for (let t = 0; t < 40; t++) {
      const a = (rng() * nU) | 0;
      const b = (rng() * nU) | 0;
      if (a === b) continue;
      const d = dot(fu, a * dim, fu, b * dim, dim);
      const na = Math.sqrt(dot(fu, a * dim, fu, a * dim, dim));
      const nb = Math.sqrt(dot(fu, b * dim, fu, b * dim, dim));
      if (na < 1e-6 || nb < 1e-6) continue;
      smooth += d / (na * nb);
      pairs++;
    }
  }
  const alphas = Array.from(layerWeights(pack));
  return {
    all: finish(all),
    cold: finish(cold),
    warm: finish(warm),
    tail: finish(tailS),
    head: finish(headS),
    smoothness: pairs ? smooth / pairs : 0,
    coverage: coverN / nI,
    alphas,
    examples,
  };
}

function knnScores(pack: Packed): { fu: Float32Array; fi: Float32Array } {
  const { nU, nI, itemDeg, trainByUser } = pack;
  const sim = new Float32Array(nI * nI);
  for (let u = 0; u < nU; u++) {
    const items = trainByUser[u];
    for (let a = 0; a < items.length; a++) {
      for (let b = 0; b < items.length; b++) {
        if (a === b) continue;
        sim[items[a] * nI + items[b]] += 1;
      }
    }
  }
  for (let a = 0; a < nI; a++) {
    for (let b = 0; b < nI; b++) {
      const denom = Math.sqrt(itemDeg[a] * itemDeg[b]);
      if (denom > 0) sim[a * nI + b] /= denom;
    }
  }
  const scores = empty(nU * nI);
  for (let u = 0; u < nU; u++) {
    const items = trainByUser[u];
    const base = u * nI;
    for (const j of items) {
      const row = j * nI;
      for (let i = 0; i < nI; i++) scores[base + i] += sim[row + i];
    }
  }
  return { fu: empty(0), fi: scores };
}

export function evaluate(session: Session): Metrics {
  const pack = sessions.get(session);
  if (!pack) throw new Error("Session is closed.");
  if (pack.cfg.model !== "knn") materializeItems(pack);
  const metrics = evaluatePack(pack);
  pack.metrics = metrics;
  pack.recall.push(metrics.all.recall);
  session.metrics = metrics;
  session.recall = pack.recall;
  session.loss = pack.loss;
  session.epoch = pack.epoch;
  return metrics;
}

export function recommendUser(session: Session, user: number, k = 8): { item: number; score: number }[] {
  const pack = sessions.get(session);
  if (!pack) throw new Error("Session is closed.");
  if (pack.cfg.model !== "knn") materializeItems(pack);
  const { nI, dim, seen } = pack;
  const scores = empty(nI);
  if (pack.cfg.model === "knn") {
    const knn = knnScores(pack);
    scores.set(knn.fi.subarray(user * nI, user * nI + nI));
  } else {
    const fwd = forward(pack);
    const { fu, fi } = combine(pack, fwd, layerWeights(pack));
    const ub = user * dim;
    for (let i = 0; i < nI; i++) scores[i] = dot(fu, ub, fi, i * dim, dim);
  }
  const ranked = topK(scores, seen, user, nI, k);
  const picks: { item: number; score: number }[] = [];
  for (let r = 0; r < ranked.length; r++) picks.push({ item: ranked[r], score: scores[ranked[r]] });
  return picks;
}

export function defaultConfig(): TrainConfig {
  return {
    model: "lightgcn",
    layers: 3,
    combine: "uniform",
    selfLoops: false,
    dim: 16,
    reg: 1e-4,
    lr: 0.02,
    epochs: 24,
    seed: 2020,
  };
}
