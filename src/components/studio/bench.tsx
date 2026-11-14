import { useMemo, useState, type ReactNode } from "react";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import {
  DATA,
  createSession,
  defaultConfig,
  evaluate,
  trainEpoch,
  type Combine,
  type Metrics,
  type ModelName,
  type TrainConfig,
} from "@/lib/rec/engine";
import { CopyButton } from "./copy";

type Row = { label: string; metrics: Metrics };

const MODELS: { id: ModelName; label: string }[] = [
  { id: "lightgcn", label: "LightGCN" },
  { id: "bpr", label: "BPR" },
  { id: "knn", label: "ItemKNN" },
  { id: "side", label: "Genre fusion" },
  { id: "ngcf", label: "NGCF-lite" },
];

function wait() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

async function fit(cfg: TrainConfig): Promise<Metrics> {
  const session = createSession(DATA, cfg);
  const epochs = cfg.model === "knn" ? 1 : cfg.epochs;
  for (let epoch = 0; epoch < epochs; epoch++) {
    trainEpoch(session);
    if (epoch === epochs - 1) return evaluate(session);
    await wait();
  }
  return evaluate(session);
}

function claim(rows: Row[]) {
  const by = (name: string) => rows.find((row) => row.label === name)?.metrics;
  const light = by("LightGCN K=3");
  const bpr = by("BPR (K=0)");
  const last = by("Last layer only");
  const side = by("Genre fusion");
  if (!light || !bpr || !last || !side) return "";
  return [
    "Pilot only — MovieLens subset, 66 users, cold cohort truncated to 3 training edges. Not ml-1m.",
    `Uniform LightGCN K=3 Recall@10 ${light.all.recall.toFixed(3)} (cold ${light.cold.recall.toFixed(3)}) vs BPR ${bpr.all.recall.toFixed(3)}.`,
    `Last-layer-only user cosine ${last.smoothness.toFixed(2)} vs uniform ${light.smoothness.toFixed(2)}.`,
    `Genre fusion tail Recall@10 ${side.tail.recall.toFixed(3)} vs LightGCN ${light.tail.recall.toFixed(3)}.`,
    "Replace this sentence with the RecBole ml-1m result before the Week 12 talk.",
  ].join(" ");
}

export function Bench() {
  const base = defaultConfig();
  const [model, setModel] = useState<ModelName>(base.model);
  const [layers, setLayers] = useState(base.layers);
  const [combine, setCombine] = useState<Combine>(base.combine);
  const [selfLoops, setSelfLoops] = useState(base.selfLoops);
  const [dim, setDim] = useState(base.dim);
  const [busy, setBusy] = useState("");
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [rows, setRows] = useState<Row[]>([]);

  const cfg = useMemo<TrainConfig>(
    () => ({
      ...base,
      model,
      layers: model === "bpr" || model === "knn" ? 0 : layers,
      combine,
      selfLoops: model === "ngcf" ? false : selfLoops,
      dim,
      lr: model === "ngcf" ? 0.01 : base.lr,
      epochs: 16,
    }),
    [model, layers, combine, selfLoops, dim, base],
  );

  async function runOne() {
    setBusy("Training");
    setMetrics(null);
    const result = await fit(cfg);
    setMetrics(result);
    setBusy("");
  }

  async function runGrid() {
    const grid: { label: string; cfg: TrainConfig }[] = [
      { label: "BPR (K=0)", cfg: { ...cfg, model: "bpr", layers: 0, epochs: 12 } },
      { label: "ItemKNN", cfg: { ...cfg, model: "knn", layers: 0, epochs: 1 } },
      { label: "LightGCN K=1", cfg: { ...cfg, model: "lightgcn", layers: 1, combine: "uniform", selfLoops: false, epochs: 12 } },
      { label: "LightGCN K=2", cfg: { ...cfg, model: "lightgcn", layers: 2, combine: "uniform", selfLoops: false, epochs: 12 } },
      { label: "LightGCN K=3", cfg: { ...cfg, model: "lightgcn", layers: 3, combine: "uniform", selfLoops: false, epochs: 12 } },
      { label: "LightGCN K=4", cfg: { ...cfg, model: "lightgcn", layers: 4, combine: "uniform", selfLoops: false, epochs: 12 } },
      { label: "Last layer only", cfg: { ...cfg, model: "lightgcn", layers: 3, combine: "last", selfLoops: false, epochs: 12 } },
      { label: "Self-loops", cfg: { ...cfg, model: "lightgcn", layers: 3, combine: "uniform", selfLoops: true, epochs: 12 } },
      { label: "Learned mix", cfg: { ...cfg, model: "lightgcn", layers: 3, combine: "learned", selfLoops: false, epochs: 12 } },
      { label: "Genre fusion", cfg: { ...cfg, model: "side", layers: 3, combine: "uniform", selfLoops: false, epochs: 12 } },
      { label: "NGCF-lite", cfg: { ...cfg, model: "ngcf", layers: 3, selfLoops: false, lr: 0.01, epochs: 12 } },
    ];
    const next: Row[] = [];
    for (const arm of grid) {
      setBusy(arm.label);
      next.push({ label: arm.label, metrics: await fit(arm.cfg) });
      setRows([...next]);
    }
    setBusy("");
  }

  const chart = metrics
    ? [
        { name: "All", recall: Number(metrics.all.recall.toFixed(3)) },
        { name: "Cold", recall: Number(metrics.cold.recall.toFixed(3)) },
        { name: "Warm", recall: Number(metrics.warm.recall.toFixed(3)) },
        { name: "Tail", recall: Number(metrics.tail.recall.toFixed(3)) },
      ]
    : [];

  const gridChart = rows.map((row) => ({
    name: row.label.replace("LightGCN ", ""),
    recall: Number(row.metrics.all.recall.toFixed(3)),
    cold: Number(row.metrics.cold.recall.toFixed(3)),
  }));

  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-md border border-line bg-surface p-4">
        <p className="text-xs font-semibold tracking-wide text-copper uppercase">Same update as RecBole LightGCN</p>
        <h2 className="font-display mt-1 text-2xl leading-tight font-semibold">A pilot you can run in the browser</h2>
        <p className="mt-2 text-sm leading-relaxed text-muted">
          {DATA.users.length} users, {DATA.items.length} films, ratings at least 4, from MovieLens latest-small.
          Eighteen users are truncated to 3 training edges so the cold slice exists. This is not ml-1m and it is not
          the number that goes on the graded slide without the word pilot.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="Model">
          <div className="flex flex-wrap gap-2">
            {MODELS.map((item) => (
              <Chip key={item.id} on={model === item.id} onClick={() => setModel(item.id)}>
                {item.label}
              </Chip>
            ))}
          </div>
        </Field>
        <Field label="Layers">
          <div className="flex flex-wrap gap-2">
            {[1, 2, 3, 4].map((k) => (
              <Chip key={k} on={layers === k} onClick={() => setLayers(k)}>
                {k}
              </Chip>
            ))}
          </div>
        </Field>
        <Field label="Layer mix">
          <div className="flex flex-wrap gap-2">
            {(["uniform", "last", "learned"] as Combine[]).map((item) => (
              <Chip key={item} on={combine === item} onClick={() => setCombine(item)}>
                {item}
              </Chip>
            ))}
          </div>
        </Field>
        <Field label="Width and self-loops">
          <div className="flex flex-wrap gap-2">
            {[8, 16, 32].map((value) => (
              <Chip key={value} on={dim === value} onClick={() => setDim(value)}>
                {value}d
              </Chip>
            ))}
            <Chip on={selfLoops} onClick={() => setSelfLoops((v) => !v)}>
              {selfLoops ? "Self-loops on" : "Self-loops off"}
            </Chip>
          </div>
        </Field>
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          disabled={busy !== ""}
          onClick={runOne}
          className="h-12 rounded-md bg-copper px-4 text-sm font-semibold text-copper-ink disabled:opacity-50"
        >
          {busy === "Training" ? "Training…" : "Train this setting"}
        </button>
        <button
          type="button"
          disabled={busy !== ""}
          onClick={runGrid}
          className="h-12 rounded-md border border-line bg-surface px-4 text-sm font-semibold disabled:opacity-50"
        >
          {busy && busy !== "Training" ? busy : "Run the proposal sweep"}
        </button>
      </div>

      {metrics ? (
        <section className="rounded-md border border-line bg-surface p-4">
          <h3 className="font-display text-lg font-semibold">Recall@10 on this subset</h3>
          <div className="mt-3 h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chart}>
                <XAxis dataKey="name" stroke="var(--color-muted)" fontSize={12} />
                <YAxis stroke="var(--color-muted)" fontSize={12} width={32} />
                <Tooltip />
                <Bar dataKey="recall" fill="var(--color-copper)" radius={4} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <dl className="mt-3 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
            <Stat label="NDCG@10" value={metrics.all.ndcg} />
            <Stat label="Hit@10" value={metrics.all.hr} />
            <Stat label="Cold Recall" value={metrics.cold.recall} />
            <Stat label="User cosine" value={metrics.smoothness} />
          </dl>
          <p className="mt-3 text-sm text-muted">
            Layer weights: {metrics.alphas.map((a) => a.toFixed(2)).join(" · ") || "n/a"}
          </p>
          <ul className="mt-3 space-y-3">
            {metrics.examples.map((example) => (
              <li key={example.user}>
                <p className="text-sm font-semibold">
                  User {DATA.users[example.user]} · {example.cold ? "cold cohort" : "warm"}
                </p>
                <p className="text-sm text-muted">
                  {example.picks.map((pick) => DATA.items[pick.item].t).join(" · ")}
                </p>
              </li>
            ))}
          </ul>
        </section>
      ) : (
        <p className="text-sm text-muted">Train once for a single setting, or run the sweep for the pilot slide.</p>
      )}

      {rows.length > 0 ? (
        <section className="rounded-md border border-line bg-surface p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="font-display text-lg font-semibold">Sweep · still a pilot</h3>
            <CopyButton text={claim(rows)} label="Copy pilot claim" />
          </div>
          <div className="mt-3 h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={gridChart}>
                <XAxis dataKey="name" stroke="var(--color-muted)" fontSize={10} interval={0} angle={-30} height={70} textAnchor="end" />
                <YAxis stroke="var(--color-muted)" fontSize={12} width={32} />
                <Tooltip />
                <Bar dataKey="recall" fill="var(--color-copper)" radius={4} />
                <Bar dataKey="cold" fill="var(--color-paper)" radius={4} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-2 text-sm leading-relaxed text-muted">{claim(rows)}</p>
        </section>
      ) : null}
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <fieldset className="rounded-md border border-line bg-surface p-3">
      <legend className="px-1 text-xs font-semibold tracking-wide text-muted uppercase">{label}</legend>
      {children}
    </fieldset>
  );
}

function Chip({ on, onClick, children }: { on: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={
        "h-11 rounded-md px-3 text-sm font-semibold " +
        (on ? "bg-copper text-copper-ink" : "border border-line bg-bg text-fg")
      }
    >
      {children}
    </button>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <dt className="text-xs text-muted">{label}</dt>
      <dd className="font-display text-xl font-semibold">{value.toFixed(3)}</dd>
    </div>
  );
}
