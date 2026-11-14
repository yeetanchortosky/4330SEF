import { useMemo, useState } from "react";
import {
  DATA,
  createSession,
  defaultConfig,
  evaluate,
  recommendUser,
  trainEpoch,
  type Metrics,
  type Session,
  type Slice,
  type TrainConfig,
} from "@/lib/rec/engine";

type Arm = {
  id: string;
  label: string;
  cfg: Partial<TrainConfig>;
  epochs: number;
};

type Row = { id: string; label: string; metrics: Metrics };

const STEPS = [
  { n: "1", title: "Data", ask: "What are we ranking, and who counts as cold?" },
  { n: "2", title: "Baselines", ask: "Does a graph beat memory and a plain embedding?" },
  { n: "3", title: "Graph", ask: "Does dropping NGCF’s extra machinery help?" },
  { n: "4", title: "Ablation", ask: "Which single LightGCN choice actually moves the number?" },
  { n: "5", title: "Cold", ask: "What happens at five interactions or fewer?" },
  { n: "6", title: "Extension", ask: "Does one small change help that slice?" },
];

const BASELINES: Arm[] = [
  { id: "bpr", label: "BPR", cfg: { model: "bpr", layers: 0 }, epochs: 12 },
  { id: "knn", label: "ItemKNN", cfg: { model: "knn", layers: 0 }, epochs: 1 },
];

const GRAPH: Arm[] = [
  { id: "light", label: "LightGCN K=3", cfg: { model: "lightgcn", layers: 3, combine: "uniform", selfLoops: false }, epochs: 12 },
  { id: "ngcf", label: "NGCF-lite K=3", cfg: { model: "ngcf", layers: 3, lr: 0.01, selfLoops: false }, epochs: 12 },
];

const ABLATION: Arm[] = [
  { id: "k1", label: "1 layer", cfg: { model: "lightgcn", layers: 1, combine: "uniform", selfLoops: false }, epochs: 10 },
  { id: "k2", label: "2 layers", cfg: { model: "lightgcn", layers: 2, combine: "uniform", selfLoops: false }, epochs: 10 },
  { id: "k4", label: "4 layers", cfg: { model: "lightgcn", layers: 4, combine: "uniform", selfLoops: false }, epochs: 10 },
  { id: "last", label: "Last layer only", cfg: { model: "lightgcn", layers: 3, combine: "last", selfLoops: false }, epochs: 10 },
  { id: "loop", label: "Self-loops", cfg: { model: "lightgcn", layers: 3, combine: "uniform", selfLoops: true }, epochs: 10 },
  { id: "d32", label: "Dim 32", cfg: { model: "lightgcn", layers: 3, dim: 32 }, epochs: 10 },
  { id: "d64", label: "Dim 64", cfg: { model: "lightgcn", layers: 3, dim: 64 }, epochs: 10 },
  { id: "regL", label: "L2 1e-5", cfg: { model: "lightgcn", layers: 3, reg: 1e-5 }, epochs: 10 },
  { id: "regH", label: "L2 1e-3", cfg: { model: "lightgcn", layers: 3, reg: 1e-3 }, epochs: 10 },
];

const EXTENSIONS: Arm[] = [
  { id: "learned", label: "Learned mix", cfg: { model: "lightgcn", layers: 3, combine: "learned", selfLoops: false }, epochs: 12 },
  { id: "side", label: "Genre fusion", cfg: { model: "side", layers: 3, combine: "uniform", selfLoops: false }, epochs: 12 },
];

function wait() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

async function fit(arm: Arm, seed: number): Promise<{ metrics: Metrics; session: Session }> {
  const cfg: TrainConfig = { ...defaultConfig(), seed, epochs: arm.epochs, ...arm.cfg };
  const session = createSession(DATA, cfg);
  const epochs = cfg.model === "knn" ? 1 : arm.epochs;
  for (let epoch = 0; epoch < epochs; epoch++) {
    trainEpoch(session);
    if (epoch < epochs - 1) await wait();
  }
  return { metrics: evaluate(session), session };
}

function n3(n: number) {
  return n.toFixed(3);
}

function meanStd(xs: number[]) {
  const mean = xs.reduce((sum, x) => sum + x, 0) / xs.length;
  if (xs.length < 2) return { mean, std: 0 };
  const v = xs.reduce((sum, x) => sum + (x - mean) ** 2, 0) / (xs.length - 1);
  return { mean, std: Math.sqrt(v) };
}

export function Study() {
  const [step, setStep] = useState(0);
  const [busy, setBusy] = useState("");
  const [rows, setRows] = useState<Record<string, Row>>({});
  const [seeds, setSeeds] = useState<{ label: string; all: number[]; cold: number[] }[]>([]);
  const [light, setLight] = useState<Session | null>(null);
  const [user, setUser] = useState(0);

  const facts = useMemo(() => {
    const nU = DATA.users.length;
    const nI = DATA.items.length;
    const cold = DATA.roles.filter((role) => role === 1).length;
    let tail = 0;
    const deg = new Int32Array(nI);
    for (const [, item] of DATA.train) deg[item]++;
    for (let i = 0; i < nI; i++) if (deg[i] <= 5) tail++;
    const sparsity = 1 - DATA.train.length / (nU * nI);
    return { nU, nI, cold, tail, sparsity, train: DATA.train.length, test: DATA.test.length };
  }, []);

  async function runList(arms: Arm[], keepLight = false) {
    for (const arm of arms) {
      setBusy(arm.label);
      const { metrics, session } = await fit(arm, 2020);
      setRows((prev) => ({ ...prev, [arm.id]: { id: arm.id, label: arm.label, metrics } }));
      if (keepLight && arm.id === "light") setLight(session);
    }
    setBusy("");
  }

  async function runSeeds() {
    const arms = [BASELINES[0], GRAPH[0]];
    const next: { label: string; all: number[]; cold: number[] }[] = [];
    for (const arm of arms) {
      const all: number[] = [];
      const cold: number[] = [];
      for (const seed of [2020, 2021, 2022]) {
        setBusy(`${arm.label} · seed ${seed}`);
        const { metrics, session } = await fit(arm, seed);
        all.push(metrics.all.recall);
        cold.push(metrics.cold.recall);
        if (arm.id === "light" && seed === 2020) {
          setLight(session);
          setRows((prev) => ({ ...prev, light: { id: "light", label: arm.label, metrics } }));
        }
        if (arm.id === "bpr" && seed === 2020) {
          setRows((prev) => ({ ...prev, bpr: { id: "bpr", label: arm.label, metrics } }));
        }
      }
      next.push({ label: arm.label, all, cold });
    }
    setSeeds(next);
    setBusy("");
  }

  const picks = light ? recommendUser(light, user, 8) : [];
  const trainCount = DATA.train.filter(([u]) => u === user).length;

  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-md border border-line bg-surface p-4">
        <p className="text-xs font-semibold tracking-wide text-copper uppercase">Proposal section 3 · six steps</p>
        <h2 className="font-display mt-1 text-2xl leading-tight font-semibold">
          High-order connectivity, then the users with almost no history.
        </h2>
        <p className="mt-2 text-sm leading-relaxed text-muted">
          Same split, same full-sort metrics, one factor at a time. This run is the MovieLens subset. The graded table
          is still RecBole on ml-1m. Numbers here are a pilot.
        </p>
      </div>

      <div className="flex gap-2 overflow-x-auto pb-1">
        {STEPS.map((item, index) => (
          <button
            key={item.n}
            type="button"
            onClick={() => setStep(index)}
            className={
              "h-11 shrink-0 rounded-md px-3 text-sm font-semibold " +
              (step === index ? "bg-copper text-copper-ink" : "border border-line bg-surface text-muted")
            }
          >
            {item.n} {item.title}
          </button>
        ))}
      </div>

      <p className="font-display text-lg leading-snug font-semibold">{STEPS[step].ask}</p>
      {busy ? <p className="text-sm text-copper">Running {busy}</p> : null}

      {step === 0 ? (
        <section className="rounded-md border border-line bg-surface p-4 text-sm leading-relaxed">
          <p>{DATA.source}</p>
          <dl className="mt-3 grid grid-cols-2 gap-3">
            <Fact label="Users" value={String(facts.nU)} />
            <Fact label="Films" value={String(facts.nI)} />
            <Fact label="Train edges" value={String(facts.train)} />
            <Fact label="Test edges" value={String(facts.test)} />
            <Fact label="Empty cells" value={`${(facts.sparsity * 100).toFixed(1)}%`} />
            <Fact label="Cold users" value={String(facts.cold)} />
            <Fact label="Tail films" value={String(facts.tail)} />
            <Fact label="Genres" value={String(DATA.genreNames.length)} />
          </dl>
          <p className="mt-3 text-muted">
            Cold, from the proposal: a user or an item with at most five training interactions. On this subset the
            cold cohort was truncated to three edges so the slice is not empty. Ratings kept here are 4 and 5. The
            ml-1m run keeps ratings of 3 and above.
          </p>
        </section>
      ) : null}

      {step === 1 ? (
        <section className="flex flex-col gap-3">
          <p className="text-sm text-muted">Lo Kam Hung’s baselines. ItemKNN is neighbors. BPR is layer 0, no graph.</p>
          <Run disabled={busy !== ""} onClick={() => runList(BASELINES)} label="Run baselines" />
          <ModelTable ids={["bpr", "knn"]} rows={rows} />
        </section>
      ) : null}

      {step === 2 ? (
        <section className="flex flex-col gap-3">
          <p className="text-sm text-muted">
            Man Timothy’s pair. NGCF-lite keeps a transform and a leaky activation. LightGCN does not. Three seeds
            are the proposal’s spread, not a second dataset.
          </p>
          <div className="flex flex-wrap gap-2">
            <Run disabled={busy !== ""} onClick={() => runList(GRAPH, true)} label="Run both once" />
            <Run disabled={busy !== ""} onClick={runSeeds} label="Repeat on 3 seeds" />
          </div>
          <ModelTable ids={["light", "ngcf"]} rows={rows} />
          {seeds.length > 0 ? (
            <div className="overflow-x-auto rounded-md border border-line">
              <table className="w-full text-left text-sm">
                <thead className="text-xs text-muted">
                  <tr>
                    <th className="px-3 py-2 font-semibold">Model</th>
                    <th className="px-3 py-2 font-semibold">Recall@10</th>
                    <th className="px-3 py-2 font-semibold">Cold Recall@10</th>
                  </tr>
                </thead>
                <tbody>
                  {seeds.map((row) => {
                    const all = meanStd(row.all);
                    const cold = meanStd(row.cold);
                    return (
                      <tr key={row.label} className="border-t border-line">
                        <td className="px-3 py-2">{row.label}</td>
                        <td className="px-3 py-2">{n3(all.mean)} ± {n3(all.std)}</td>
                        <td className="px-3 py-2">{n3(cold.mean)} ± {n3(cold.std)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : null}
          <div className="rounded-md border border-line bg-surface p-4">
            <label className="text-xs font-semibold tracking-wide text-muted uppercase" htmlFor="who">
              Rank a user with the LightGCN checkpoint
            </label>
            <select
              id="who"
              value={user}
              onChange={(event) => setUser(Number(event.target.value))}
              className="mt-2 h-12 w-full rounded-md border border-line bg-bg px-3 text-base"
            >
              {DATA.users.map((name, index) => (
                <option key={name} value={index}>
                  {name} · {DATA.roles[index] === 1 ? "cold" : "warm"}
                </option>
              ))}
            </select>
            {light ? (
              <>
                <p className="mt-2 text-sm text-muted">{trainCount} training films. Seen films are held out.</p>
                <ol className="mt-2 space-y-1 text-sm">
                  {picks.map((pick, index) => (
                    <li key={pick.item}>
                      {index + 1}. {DATA.items[pick.item].t}
                    </li>
                  ))}
                </ol>
              </>
            ) : (
              <p className="mt-2 text-sm text-muted">Run LightGCN first. The list is the checkpoint, not a new model.</p>
            )}
          </div>
        </section>
      ) : null}

      {step === 3 ? (
        <section className="flex flex-col gap-3">
          <p className="text-sm text-muted">
            Proposal table: layers 1–4, uniform versus last layer, self-loops, width, L2. The reference row is
            LightGCN K=3 from step 3, dim 16, L2 1e-4, no self-loop. One factor moves.
          </p>
          <Run disabled={busy !== ""} onClick={() => runList(ABLATION)} label="Run the one-factor grid" />
          <ModelTable ids={["light", ...ABLATION.map((arm) => arm.id)]} rows={rows} />
        </section>
      ) : null}

      {step === 4 ? <Cold rows={rows} /> : null}

      {step === 5 ? (
        <section className="flex flex-col gap-3">
          <p className="text-sm text-muted">
            Two changes, trained apart. Learned mix starts uniform, so step 0 is still LightGCN. Genre fusion adds a
            linear map of categories, which a new film still has. Judge both on the cold row, not on the average.
          </p>
          <Run disabled={busy !== ""} onClick={() => runList(EXTENSIONS)} label="Run both extensions" />
          <ModelTable ids={["light", "learned", "side"]} rows={rows} />
        </section>
      ) : null}
    </div>
  );
}

function Cold({ rows }: { rows: Record<string, Row> }) {
  const light = rows.light?.metrics;
  if (!light) {
    return <p className="text-sm text-muted">Run LightGCN in step 3. This step does not retrain. It cuts that checkpoint.</p>;
  }
  const slices: { label: string; slice: Slice }[] = [
    { label: "All users", slice: light.all },
    { label: "Cold users ≤5", slice: light.cold },
    { label: "Warm users", slice: light.warm },
    { label: "Tail films ≤5", slice: light.tail },
    { label: "Head films", slice: light.head },
  ];
  const drop = light.all.recall - light.cold.recall;
  return (
    <section className="flex flex-col gap-3">
      <p className="text-sm text-muted">
        Same checkpoint. Recall drop from all users to cold users is {n3(drop)}. Coverage of the catalogue in
        someone’s top 10 is {n3(light.coverage)}. User-embedding cosine is {n3(light.smoothness)}.
      </p>
      <div className="overflow-x-auto rounded-md border border-line">
        <table className="w-full text-left text-sm">
          <thead className="text-xs text-muted">
            <tr>
              <th className="px-3 py-2 font-semibold">Slice</th>
              <th className="px-3 py-2 font-semibold">n</th>
              <th className="px-3 py-2 font-semibold">R@10</th>
              <th className="px-3 py-2 font-semibold">N@10</th>
              <th className="px-3 py-2 font-semibold">H@10</th>
              <th className="px-3 py-2 font-semibold">R@20</th>
            </tr>
          </thead>
          <tbody>
            {slices.map((row) => (
              <tr key={row.label} className="border-t border-line">
                <td className="px-3 py-2">{row.label}</td>
                <td className="px-3 py-2">{row.slice.n}</td>
                <td className="px-3 py-2">{n3(row.slice.recall)}</td>
                <td className="px-3 py-2">{n3(row.slice.ndcg)}</td>
                <td className="px-3 py-2">{n3(row.slice.hr)}</td>
                <td className="px-3 py-2">{n3(row.slice.recall20)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ModelTable({ ids, rows }: { ids: string[]; rows: Record<string, Row> }) {
  const shown = ids.map((id) => rows[id]).filter(Boolean);
  if (!shown.length) return <p className="text-sm text-muted">No run yet.</p>;
  return (
    <div className="overflow-x-auto rounded-md border border-line">
      <table className="w-full min-w-[36rem] text-left text-sm">
        <thead className="text-xs text-muted">
          <tr>
            <th className="px-3 py-2 font-semibold">Model</th>
            <th className="px-3 py-2 font-semibold">R@10</th>
            <th className="px-3 py-2 font-semibold">N@10</th>
            <th className="px-3 py-2 font-semibold">H@10</th>
            <th className="px-3 py-2 font-semibold">Cold R@10</th>
            <th className="px-3 py-2 font-semibold">Tail R@10</th>
            <th className="px-3 py-2 font-semibold">Cosine</th>
          </tr>
        </thead>
        <tbody>
          {shown.map((row) => (
            <tr key={row.id} className="border-t border-line">
              <td className="px-3 py-2">{row.label}</td>
              <td className="px-3 py-2">{n3(row.metrics.all.recall)}</td>
              <td className="px-3 py-2">{n3(row.metrics.all.ndcg)}</td>
              <td className="px-3 py-2">{n3(row.metrics.all.hr)}</td>
              <td className="px-3 py-2">{n3(row.metrics.cold.recall)}</td>
              <td className="px-3 py-2">{n3(row.metrics.tail.recall)}</td>
              <td className="px-3 py-2">{n3(row.metrics.smoothness)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Run({ label, onClick, disabled }: { label: string; onClick: () => void; disabled: boolean }) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className="h-12 rounded-md bg-copper px-4 text-sm font-semibold text-copper-ink disabled:opacity-50"
    >
      {label}
    </button>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs text-muted">{label}</dt>
      <dd className="font-display text-xl font-semibold">{value}</dd>
    </div>
  );
}
