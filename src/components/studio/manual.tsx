import { type ReactNode } from "react";
import { ADAPTIVE_PY, BASE_YAML, LIGHTGCN_YAML, OWNERS, SIDE_PY } from "@/lib/research/protocol";
import { Atlas } from "./atlas";
import { CopyButton } from "./copy";

const BLOCKS = [
  { file: "config/base.yaml", body: BASE_YAML },
  { file: "config/lightgcn.yaml", body: LIGHTGCN_YAML },
  { file: "model/lightgcn_adaptive.py", body: ADAPTIVE_PY },
  { file: "model/lightgcn_side.py", body: SIDE_PY },
];

export function Manual() {
  return (
    <article className="flex flex-col gap-4 text-sm leading-relaxed">
      <header className="rounded-md border border-line bg-surface p-4">
        <p className="text-xs font-semibold tracking-wide text-copper uppercase">Operator’s manual</p>
        <h2 className="font-display mt-1 text-2xl leading-tight font-semibold">How to use Orderline</h2>
        <p className="mt-2 text-muted">
          Orderline is the instrument for the Group 5 proposal, Graph-Based Recommendation in RecBole. It runs that
          experiment in the browser and holds the notes for the six-minute talk. It does not replace the RecBole
          table.
        </p>
      </header>

      <Section title="1. What you are looking at">
        <p>
          The proposal’s procedure is six steps: prepare the data, run the baselines, reproduce LightGCN against
          NGCF, ablate one factor, cut the result at five interactions, then try one small extension. The Study tab
          is those six steps, in that order. Lab is the same trainer with the knobs free. Slides is the Week 11
          talk. This page is the manual.
        </p>
      </Section>

      <Section title="2. What a number on this screen is not">
        <ul className="list-disc space-y-2 pl-5 text-muted">
          <li>It is not MovieLens-1M. The subset is 66 users from MovieLens latest-small.</li>
          <li>Ratings kept here are 4 and 5. The graded ml-1m run keeps 3 and above.</li>
          <li>Eighteen users were cut to three training films so a cold slice exists. Say that out loud.</li>
          <li>NeuMF and SASRec are not in this browser core. They are RecBole runs, owned by Yeung and Tam.</li>
          <li>A second catalogue (Amazon-Beauty or Gowalla) is a later RecBole run, not a button here.</li>
        </ul>
      </Section>

      <Section title="3. Study, step by step">
        <ol className="list-decimal space-y-2 pl-5">
          <li>
            <span className="font-semibold">Data. </span>
            Read the counts. Cold means at most five training edges, which is the definition in proposal section 3.6.
            Do not move it after you see a chart.
          </li>
          <li>
            <span className="font-semibold">Baselines. </span>
            ItemKNN and BPR. BPR is the embedding with no propagation. If LightGCN cannot beat it, the graph is not
            earning its place.
          </li>
          <li>
            <span className="font-semibold">Graph. </span>
            LightGCN against the NGCF-lite. “Repeat on 3 seeds” is the proposal’s mean ± spread. Then pick a user.
            The list is that checkpoint. Films they already rated are held out.
          </li>
          <li>
            <span className="font-semibold">Ablation. </span>
            Layers, last-layer-only, self-loops, width, L2. One factor moves. The reference is LightGCN with 3
            layers, width 16, L2 of 1e-4, no self-loop. On ml-1m the widths are 32, 64, and 128. Width 16 is the
            subset stand-in, not a silent substitution.
          </li>
          <li>
            <span className="font-semibold">Cold. </span>
            No new training. The same LightGCN checkpoint, cut into all, cold, warm, tail, and head. The drop is
            all-user Recall@10 minus cold Recall@10.
          </li>
          <li>
            <span className="font-semibold">Extension. </span>
            Learned layer mix, then genre fusion, never both in the first table. Read the cold column and the tail
            column. A null result is a result.
          </li>
        </ol>
      </Section>

      <Section title="4. How to read a cell">
        <ul className="list-disc space-y-2 pl-5 text-muted">
          <li>Recall@K divides hits by how many relevant films that user had in the test set.</li>
          <li>NDCG@K discounts a hit by how far down the list it sits. Models are chosen on NDCG@10.</li>
          <li>Hit@10 is 1 if any relevant film landed in the ten.</li>
          <li>Full sort: every unseen film is ranked. Sampled metrics are not used.</li>
          <li>Cosine near 1 means user vectors have collapsed. That is the over-smoothing check.</li>
          <li>Coverage is the share of the catalogue that appears in at least one top 10.</li>
          <li>If a gap is inside the three-seed spread, write tie. Do not pick the larger mean.</li>
        </ul>
      </Section>

      <Section title="5. Who owns which piece">
        <ul className="divide-y divide-line">
          {OWNERS.map((owner) => (
            <li key={owner.name} className="py-2">
              <span className="font-semibold">{owner.name}. </span>
              <span className="text-muted">{owner.owns}. </span>
              <span className="text-copper">{owner.file}</span>
            </li>
          ))}
        </ul>
        <p className="mt-2 text-muted">
          Tam Ho Chung is group leader, submits the one zip, and owns the cold slice. The report file is named
          SessionP08_Group5 Final Report.pdf. All five people are in the room for the talk.
        </p>
      </Section>

      <Section title="6. The graded run, outside this browser">
        <p className="text-muted">
          Install RecBole, keep one shared config, and paste the two subclasses. Do not fork LightGCN. Week-1 gate
          is ml-100k. The table in the report is ml-1m, full sort, three seeds. Amazon-Beauty is the second set
          because it has categories. Gowalla does not, so it cannot test genre fusion.
        </p>
        <div className="mt-3 flex flex-col gap-3">
          {BLOCKS.map((block) => (
            <div key={block.file}>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="text-xs font-semibold text-copper">{block.file}</p>
                <CopyButton text={block.body} label="Copy" />
              </div>
              <pre className="mt-2 overflow-x-auto rounded-md bg-bg p-3 text-xs leading-relaxed">{block.body}</pre>
            </div>
          ))}
        </div>
      </Section>

      <Section title="7. The talk">
        <p className="text-muted">
          Six minutes, plus two or three of questions. Initial slides are due 15 Nov 2026, 23:59. The Slides tab is
          already cut to 5.5 minutes. Do not put a subset number on a slide without the word pilot. Time management
          is 10% of the presentation, and the presentation is 70% of the project.
        </p>
      </Section>

      <Section title="8. Decisions behind the build">
        <p className="mb-3 text-muted">
          Each card is a paper finding and the choice it forced. The proposal text is not rewritten here.
        </p>
        <Atlas />
      </Section>
    </article>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="rounded-md border border-line bg-surface p-4">
      <h3 className="font-display text-lg font-semibold">{title}</h3>
      <div className="mt-2">{children}</div>
    </section>
  );
}
