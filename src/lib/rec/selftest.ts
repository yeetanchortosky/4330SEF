import { DATA, createSession, defaultConfig, evaluate, trainEpoch } from "./engine";

function run(label: string, model: "bpr" | "lightgcn" | "side" | "ngcf" | "knn", extra: Partial<ReturnType<typeof defaultConfig>> = {}) {
  const cfg = { ...defaultConfig(), model, epochs: 12, ...extra };
  const session = createSession(DATA, cfg);
  if (model === "knn") {
    const m = evaluate(session);
    console.log(label, "recall", m.all.recall.toFixed(3), "cold", m.cold.recall.toFixed(3), "n", m.cold.n);
    return m;
  }
  let first = 0;
  for (let e = 0; e < cfg.epochs; e++) {
    const loss = trainEpoch(session);
    if (e === 0) first = loss;
    if (!Number.isFinite(loss)) throw new Error(label + " nonfinite loss " + loss);
  }
  const m = evaluate(session);
  const last = session.loss[session.loss.length - 1];
  console.log(
    label,
    "loss",
    first.toFixed(3),
    "→",
    last.toFixed(3),
    "recall",
    m.all.recall.toFixed(3),
    "ndcg",
    m.all.ndcg.toFixed(3),
    "cold",
    m.cold.recall.toFixed(3),
    "warm",
    m.warm.recall.toFixed(3),
    "tail",
    m.tail.recall.toFixed(3),
    "smooth",
    m.smoothness.toFixed(3),
    "alpha",
    m.alphas.map((a) => a.toFixed(2)).join(","),
  );
  if (!(last < first * 0.9)) throw new Error(label + " did not learn");
  if (!(m.all.recall > 0.04)) throw new Error(label + " recall too low " + m.all.recall);
  return m;
}

const t0 = Date.now();
run("bpr", "bpr");
run("light", "lightgcn");
run("side", "side");
run("ngcf", "ngcf", { lr: 0.01 });
run("last", "lightgcn", { combine: "last" });
run("loop", "lightgcn", { selfLoops: true });
run("learned", "lightgcn", { combine: "learned" });
run("knn", "knn");
console.log("ms", Date.now() - t0);
console.log("users", DATA.users.length, "items", DATA.items.length, "cold", DATA.roles.filter((r) => r === 1).length);
