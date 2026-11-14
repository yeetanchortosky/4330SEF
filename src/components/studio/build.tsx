import { ADAPTIVE_PY, BASE_YAML, COLD_STEPS, HYPER_TEST, LIGHTGCN_YAML, OWNERS, RUN_PY, SIDE_PY } from "@/lib/research/protocol";
import { CopyButton } from "./copy";

const BLOCKS = [
  { title: "Shared config", file: "config/base.yaml", body: BASE_YAML },
  { title: "LightGCN point", file: "config/lightgcn.yaml", body: LIGHTGCN_YAML },
  { title: "One-factor search", file: "hyper.test", body: HYPER_TEST },
  { title: "Learned layer mix", file: "model/lightgcn_adaptive.py", body: ADAPTIVE_PY },
  { title: "Genre prior", file: "model/lightgcn_side.py", body: SIDE_PY },
  { title: "First run", file: "run.py", body: RUN_PY },
];

export function Build() {
  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-md border border-line bg-surface p-4">
        <p className="text-xs font-semibold tracking-wide text-copper uppercase">Code the brief asked for</p>
        <h2 className="font-display mt-1 text-2xl leading-tight font-semibold">RecBole classes, not a second proposal</h2>
        <p className="mt-2 text-sm leading-relaxed text-muted">
          Week-1 gate is <span className="text-fg">ml-100k</span>. Graded table is <span className="text-fg">ml-1m</span>,
          ratings at least 3, full sort. Second set is Amazon-Beauty, because it has categories. Paste these files
          around an installed RecBole. The browser bench is the same math on a subset.
        </p>
      </div>

      <section className="rounded-md border border-line bg-surface p-4">
        <h3 className="font-display text-lg font-semibold">Who owns which file</h3>
        <ul className="mt-3 divide-y divide-line">
          {OWNERS.map((owner) => (
            <li key={owner.name} className="flex flex-col gap-1 py-3 text-sm sm:flex-row sm:items-baseline sm:justify-between">
              <span className="font-semibold">{owner.name}</span>
              <span className="text-muted">{owner.owns}</span>
              <span className="text-copper">{owner.file}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="rounded-md border border-line bg-surface p-4">
        <h3 className="font-display text-lg font-semibold">Cold slice, after the checkpoint</h3>
        <ol className="mt-3 list-decimal space-y-2 pl-5 text-sm leading-relaxed text-muted">
          {COLD_STEPS.map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ol>
      </section>

      <section className="rounded-md border border-line bg-surface p-4">
        <h3 className="font-display text-lg font-semibold">Report, when the numbers exist</h3>
        <p className="mt-2 text-sm leading-relaxed text-muted">
          Due 6 Dec 2026, 23:59. One PDF named SessionP08_Group5 Final Report.pdf, plus slides and code, one zip from
          Tam. At most 2,000 words, not counting references, figures, tables, work allocation, or the appendix. Late
          submission is not accepted. You can replace the file up to one day before.
        </p>
        <ul className="mt-3 space-y-2 text-sm">
          <li><span className="font-semibold">Abstract, 10%.</span> <span className="text-muted">100–150 words. The claim and the slice, not a library description.</span></li>
          <li><span className="font-semibold">Introduction and literature, 20%.</span> <span className="text-muted">The history that would not fit in six minutes.</span></li>
          <li><span className="font-semibold">Methodology and results, 60% × difficulty.</span> <span className="text-muted">The grid, the cold drop, the two extensions. A tie is allowed.</span></li>
          <li><span className="font-semibold">Conclusion and references, 10%.</span> <span className="text-muted">APA, already started in the proposal. Do not renumber them.</span></li>
        </ul>
      </section>

      {BLOCKS.map((block) => (
        <section key={block.file} className="rounded-md border border-line bg-surface p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="font-display text-lg font-semibold">{block.title}</h3>
              <p className="text-xs text-copper">{block.file}</p>
            </div>
            <CopyButton text={block.body} label="Copy" />
          </div>
          <pre className="mt-3 overflow-x-auto rounded-md bg-bg p-3 text-xs leading-relaxed text-fg">{block.body}</pre>
        </section>
      ))}
    </div>
  );
}
