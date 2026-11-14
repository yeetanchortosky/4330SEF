export type SlideVisual = "title" | "equation" | "models" | "flow" | "ablation" | "cold" | "team" | "bars" | "claim";

export type Slide = {
  id: string;
  part: "Talk" | "Backup";
  minutes: number;
  kicker: string;
  claim: string;
  lines: string[];
  visual: SlideVisual;
  speak: string;
  avoid: string;
  make: string;
};

/** Spoken talk is 5.5 min. Leave 30 seconds. Time management is 10% of the presentation mark. */
export const SLIDES: Slide[] = [
  {
    id: "01",
    part: "Talk",
    minutes: 0.4,
    kicker: "Title · 25 sec",
    claim: "We test which part of LightGCN still works when a user has at most five training edges.",
    lines: ["COMP 4330SEF · P08 · Group 5", "Open-source system: RecBole", "Initial slides · 15 Nov 2026"],
    visual: "title",
    speak:
      "One sentence, then names. We are not introducing RecBole. We are testing a claim about LightGCN. The proposal stands. This deck is the six-minute cut.",
    avoid: "Reading the abstract, listing every member, or saying 'nowadays'.",
    make: "Serif claim, 36pt, paper background, one copper rule. Course line in the footer only.",
  },
  {
    id: "02",
    part: "Talk",
    minutes: 0.9,
    kicker: "Gap · 55 sec",
    claim: "The published tables never tell you what happens under five interactions.",
    lines: [
      "ml-1m is still 96% empty after dropping ratings below 3",
      "LightGCN beats NGCF by simplifying the graph step",
      "Nobody reports that gain on the cold slice",
    ],
    visual: "bars",
    speak:
      "He et al. showed that throwing away transforms and activations beat NGCF by about sixteen percent relative recall. RecBole-GNN's own ml-1m full-sort table agrees on the sign: LightGCN 0.186, NGCF 0.181, BPR 0.178. Those numbers are everyone, mixed together. A user with three films is the case a real system hits. That slice is the project.",
    avoid: "A history from GroupLens to GPT. That is the report, and the report has a 2,000-word cap.",
    make: "Left: 96% empty bar. Right: three published Recall@10 ticks, labeled 'not our run'. Footer: He et al. 2020; RecBole-GNN ml-1m.",
  },
  {
    id: "03",
    part: "Talk",
    minutes: 1,
    kicker: "Model · 60 sec",
    claim: "LightGCN keeps neighbor averaging. It drops the rest on purpose.",
    lines: [
      "Propagate: normalized adjacency, no self-loop",
      "Combine: uniform average of layer 0 through K",
      "Score: inner product. Train: BPR. Only layer 0 is learned.",
    ],
    visual: "equation",
    speak:
      "Say the equation once. Each layer is the normalized sum of neighbors, nothing else. The vector we rank with is the average of every layer, including the raw ID embedding. That average is why a self-loop is optional: layer 0 is already in the mix. If you keep only the last layer, the vectors collapse. We will show that, not just cite it.",
    avoid: "A GCN lecture. No derivation.",
    make: "Two lines of math, 28pt. A four-node bipartite sketch labeled hop 1 and hop 2.",
  },
  {
    id: "04",
    part: "Talk",
    minutes: 1.1,
    kicker: "Experiment · 65 sec",
    claim: "One factor moves. Then we cut the same ranking by degree.",
    lines: [
      "BPR is layer 0. Then layers 1 to 4.",
      "Mix: uniform, last layer only, or learned weights",
      "Cold user or item: at most 5 training edges",
      "Full-sort Recall, NDCG, Hit at 10 and 20. Three seeds.",
    ],
    visual: "ablation",
    speak:
      "RecBole is the control: one split, one negative sampler, full-sort evaluation. Sampled metrics can reverse a ranking, so we do not use them. Cold is a threshold we froze at five, from the proposal. We do not retrain on cold users. We slice the test of the checkpoint that won on validation NDCG. A win has to clear the spread across seeds. Otherwise we call it a tie.",
    avoid: "Reading the whole ablation grid. The grid is a backup slide.",
    make: "Five factor rows. The phrase 'same checkpoint' in copper. Do not put hyperparameters in the title.",
  },
  {
    id: "05",
    part: "Talk",
    minutes: 1.2,
    kicker: "Extension · 70 sec",
    claim: "Two one-line changes, tried separately, judged on the cold slice.",
    lines: [
      "Learned mix: softmax over layers, started at uniform",
      "Genre prior: item embedding plus a linear map of its categories",
      "Pilot, not ml-1m: last-layer cosine goes to 1; genres move tail items",
    ],
    visual: "cold",
    speak:
      "The paper tried learned layer weights and left them out. We put only that back. Logits start at zero, so the first step is still LightGCN. The second change adds genres, which a new item still has when it has no edges. We do not turn both on in the first table. The bench on a MovieLens subset is a pilot: uniform LightGCN beat BPR, last-layer-only over-smoothed, and the genre map was what moved tail items. Say pilot. The ml-1m table is empty until the runs exist. You may show fifteen seconds of the bench. Not more. Time is graded.",
    avoid: "Calling the pilot MovieLens-1M. Calling a null result a failure. A null result on the cold slice is a finding.",
    make: "Left: layer-weight bars. Right: ID arrow plus genre arrow. A caption in copper: 'Pilot · 66 users · not the graded table'.",
  },
  {
    id: "06",
    part: "Talk",
    minutes: 0.9,
    kicker: "Close · 50 sec",
    claim: "Either high-order neighbors help the sparse users, or they do not. Both answers count.",
    lines: [
      "Week 11: this argument. Week 12: the ml-1m numbers.",
      "If the gap is inside the seed spread, we will say tie.",
      "Questions.",
    ],
    visual: "claim",
    speak:
      "Stop on the claim. Do not add a thank-you slide. Do not open a new idea. The report, the code, and the six-minute talk are three different lengths of the same experiment. Backup slides are for the questions, which are a fifth of the presentation mark.",
    avoid: "An LLM, a fairness detour, or a live training run that might not finish.",
    make: "The claim at 36pt. Nothing else but the word Questions and the group footer.",
  },
  {
    id: "B1",
    part: "Backup",
    minutes: 0,
    kicker: "Q&A · why RecBole",
    claim: "RecBole is the instrument. LightGCN is the object.",
    lines: [
      "Same split, sampler, and full-sort metrics, or the ablation is void",
      "A from-scratch loader would spend the term on bugs",
      "The new code is two subclasses, not a library tour",
    ],
    visual: "claim",
    speak:
      "Twenty seconds. The brief asks for analysis of an open-source system, not a reimplementation. Difficulty stays high only if the analysis is the work. A tutorial that runs one model is how a project gets marked Easy, coefficient 0.9.",
    avoid: "Apologizing for using a library.",
    make: "Three lines. Footer: brief, 'analysis rather than introduction'.",
  },
  {
    id: "B2",
    part: "Backup",
    minutes: 0,
    kicker: "Q&A · why not a newer GNN",
    claim: "XSimGCL is higher on ml-1m. It is a different question.",
    lines: [
      "Recall@10: BPR 0.178, NGCF 0.181, LightGCN 0.186",
      "SGL 0.189 · XSimGCL 0.212",
      "Contrastive loss would hide which LightGCN piece moved",
    ],
    visual: "bars",
    speak:
      "We know the ceiling. Adding SimGCL changes the objective, so the one-factor rule dies. That is future work, one sentence, not a missing baseline.",
    avoid: "Pretending LightGCN is the 2024 state of the art.",
    make: "One bar chart. Source line: RecBole-GNN ml-1m, rating ≥ 3, full sort.",
  },
  {
    id: "B3",
    part: "Backup",
    minutes: 0,
    kicker: "Q&A · is five really cold",
    claim: "Five edges is few-shot. Zero edges cannot be solved by collaborative filtering.",
    lines: [
      "Threshold frozen before we look at test numbers",
      "Same checkpoint, four slices: all, cold user, warm user, tail item",
      "Genres cover the item that has a category and no history",
    ],
    visual: "cold",
    speak:
      "If they say 'true cold start', agree, and point at the genre arm. Do not move the threshold after seeing the chart.",
    avoid: "Redefining cold as whatever looks good.",
    make: "A 2 by 2. Highlight cold user × tail item.",
  },
  {
    id: "B4",
    part: "Backup",
    minutes: 0,
    kicker: "Q&A · metrics",
    claim: "Full sort, or we are not allowed to compare with the papers.",
    lines: [
      "Hit@10 is a yes. Recall divides by how many relevant items exist.",
      "NDCG discounts a hit by its rank.",
      "We select models on validation NDCG@10. Test is touched once.",
    ],
    visual: "claim",
    speak:
      "Krichene and Rendle: sampled top-K can rank A over B even when the full ranking disagrees, and more samples do not fix the bias. That is the whole answer.",
    avoid: "A worked NDCG example unless they ask for the formula. It is the next slide.",
    make: "Three definitions, one line each.",
  },
  {
    id: "B5",
    part: "Backup",
    minutes: 0,
    kicker: "Q&A · math",
    claim: "Only the ID table is trained. The layers are a deterministic average.",
    lines: ["E(k+1) = D−1/2 A D−1/2 E(k)", "e = average of E(0) … E(K)", "Loss = −log σ(ŷui − ŷuj) + λ‖E0‖²"],
    visual: "equation",
    speak: "j is one uniform negative. Regularization hits the ego embeddings of the triple, which is what RecBole's EmbLoss does. Prediction is a dot product.",
    avoid: "Bayesian derivation of BPR.",
    make: "Three lines, large.",
  },
  {
    id: "B6",
    part: "Backup",
    minutes: 0,
    kicker: "Q&A · what we cut",
    claim: "The proposal is intact. Six minutes cannot carry it.",
    lines: [
      "Cut from the talk: GroupLens-to-GPT history, GRU4Rec, BERT4Rec, Gowalla",
      "Cut: fairness, LLMs, knowledge graphs",
      "Kept: the slice, the one-factor grid, the two extensions",
    ],
    visual: "claim",
    speak:
      "If a member asks why their baseline is missing, this is the slide. SASRec stays as the single sequential check, in the report and in Q&A, not in the six minutes.",
    avoid: "Implying the proposal was wrong. It was written for the report.",
    make: "Two columns: said / not said.",
  },
  {
    id: "B7",
    part: "Backup",
    minutes: 0,
    kicker: "Q&A · who",
    claim: "Each person owns a file. The leader also owns the cold slice.",
    lines: [
      "Lo Kam Hung — yaml, ItemKNN, BPR",
      "Yeung Kwan Ho — NeuMF and both extension classes",
      "Man Timothy Tsz Hin — LightGCN, NGCF, baseline table",
      "Tam Ho Chung — SASRec, cold slice, submission",
      "Suen Ching — runs.csv and every chart",
    ],
    visual: "team",
    speak: "All five must be in the room. Who speaks is our choice. Tam submits one zip. The report file name is SessionP08_Group5 Final Report.pdf.",
    avoid: "A Gantt chart.",
    make: "Five rows, a filename at the end of each.",
  },
  {
    id: "B8",
    part: "Backup",
    minutes: 0,
    kicker: "Q&A · dates",
    claim: "Tomorrow is the slide deadline. The numbers can still move.",
    lines: [
      "15 Nov 2026, 23:59 — initial slides on OLE. Revision before the talk is allowed.",
      "Week 12/13 — 6 min + 2 to 3 min questions. P08 is Thu 17:00, C0411, tutor Weidong.",
      "6 Dec 2026, 23:59 — report, slides, and code. Late work is not accepted.",
    ],
    visual: "flow",
    speak: "Say this only if they ask about process. Do not spend talk time on the calendar.",
    avoid: "Reading the whole brief.",
    make: "Three dates, nothing else.",
  },
  {
    id: "B9",
    part: "Backup",
    minutes: 0,
    kicker: "Backup · references",
    claim: "Five papers a question will actually come from.",
    lines: [
      "He et al., 2020, LightGCN",
      "Wang et al., 2019, NGCF",
      "Rendle et al., 2009, BPR · Krichene and Rendle, 2020, sampled metrics",
      "Zhao et al., 2021, RecBole · Wu et al., 2022, GNN survey",
    ],
    visual: "claim",
    speak: "Leave it up. Do not read it. Full APA stays in the proposal and the report.",
    avoid: "Twenty references at 11pt.",
    make: "Author, year, one label. 24pt.",
  },
];

export const TALK_MINUTES = SLIDES.filter((s) => s.part === "Talk").reduce((sum, s) => sum + s.minutes, 0);

export const DECK_RULES = [
  {
    title: "Six minutes, then stop",
    body: "The brief is 6 minutes plus 2–3 minutes of questions. Time management is 10% of the presentation, and the presentation is 70% of the project. This talk is 5.5 minutes so you have a breath.",
  },
  {
    title: "Analysis, not a tour",
    body: "The marker asked to hear your analysis of the system, not an introduction of somebody else's work. RecBole is named once. The claim is about LightGCN under five interactions.",
  },
  {
    title: "The title is a sentence",
    body: "“Methodology” is not a title. If the ml-1m runs are not in, the title stays a question or a protocol claim. Never invent a result.",
  },
  {
    title: "Pilot numbers wear a label",
    body: "The bench is a MovieLens subset with a truncated cold cohort. It is for direction. The graded table is RecBole on ml-1m, full sort.",
  },
  {
    title: "High difficulty is the slice",
    body: "A single demo that runs LightGCN can be marked Easy (×0.9). The frozen threshold, the one-factor grid, and a result you are willing to call a tie are what make it research-level (×1.0).",
  },
  {
    title: "Tomorrow, then revise",
    body: "Initial slides are due 15 Nov 2026, 23:59. You may still change them before the Week 12/13 talk. The proposal is not being rewritten.",
  },
];
