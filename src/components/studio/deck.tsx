import { useEffect, useMemo, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { DECK_RULES, SLIDES, TALK_MINUTES, type Slide } from "@/lib/research/slides";
import { CopyButton } from "./copy";

function notesFor(slide: Slide) {
  return `${slide.kicker}\n${slide.claim}\n\nSay:\n${slide.speak}\n\nDo not:\n${slide.avoid}\n\nOn the slide:\n${slide.lines.join("\n")}\n\nBuild it:\n${slide.make}`;
}

function SlideFace({ slide, index }: { slide: Slide; index: number }) {
  return (
    <article className="flex aspect-video w-full flex-col bg-paper p-4 text-ink sm:p-6">
      <p className="text-xs font-semibold tracking-wide text-copper uppercase">{slide.kicker}</p>
      <h3 className="font-display mt-2 text-lg leading-tight font-semibold sm:text-2xl">{slide.claim}</h3>
      <div className="mt-3 flex min-h-0 flex-1 flex-col justify-end gap-2">
        {slide.visual === "equation" ? (
          <p className="font-display text-sm leading-snug text-ink sm:text-base">
            E<sup>k+1</sup> = D<sup>−½</sup> A D<sup>−½</sup> E<sup>k</sup>
            <span className="mt-1 block text-sm">final = mean of layers 0…K · score = dot product</span>
          </p>
        ) : null}
        <ul className="space-y-1">
          {slide.lines.map((line) => (
            <li key={line} className="border-l-2 border-copper pl-2 text-sm leading-snug text-ink">
              {line}
            </li>
          ))}
        </ul>
      </div>
      <p className="mt-3 text-xs text-faint">
        Group 5 · P08 · {index + 1}
      </p>
    </article>
  );
}

export function Deck() {
  const [part, setPart] = useState<"Talk" | "Backup">("Talk");
  const slides = useMemo(() => SLIDES.filter((s) => s.part === part), [part]);
  const [index, setIndex] = useState(0);
  const slide = slides[Math.min(index, slides.length - 1)];

  useEffect(() => {
    setIndex(0);
  }, [part]);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const tag = (event.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if (event.key === "ArrowRight") setIndex((n) => Math.min(slides.length - 1, n + 1));
      if (event.key === "ArrowLeft") setIndex((n) => Math.max(0, n - 1));
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [slides.length]);

  const talk = SLIDES.filter((s) => s.part === "Talk");
  const markdown = talk
    .map(
      (s, i) =>
        `## ${i + 1}. ${s.claim}\n\n${s.lines.map((l) => `- ${l}`).join("\n")}\n\nNotes: ${s.speak}`,
    )
    .join("\n\n");

  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-md border border-line bg-surface p-4">
        <p className="text-xs font-semibold tracking-wide text-copper uppercase">Week 11 deck · proposal unchanged</p>
        <h2 className="font-display mt-1 text-2xl leading-tight font-semibold">
          {TALK_MINUTES.toFixed(1)} minutes spoken. Half a minute of air.
        </h2>
        <p className="mt-2 text-sm leading-relaxed text-muted">
          Initial slides are due 15 Nov 2026, 23:59. The talk itself is 6 minutes in Week 12/13, P08, Thursday 17:00,
          room C0411. Questions are another 2–3 minutes and a fifth of the presentation mark. This cut keeps the
          proposal’s experiment and drops only what cannot be said in time.
        </p>
      </div>

      <details className="rounded-md border border-line bg-surface">
        <summary className="cursor-pointer px-4 py-3 text-sm font-semibold">Six rules before you open PowerPoint</summary>
        <ul className="space-y-3 px-4 pb-4">
          {DECK_RULES.map((rule) => (
            <li key={rule.title}>
              <p className="text-sm font-semibold">{rule.title}</p>
              <p className="text-sm leading-relaxed text-muted">{rule.body}</p>
            </li>
          ))}
        </ul>
      </details>

      <div className="flex gap-2">
        {(["Talk", "Backup"] as const).map((name) => (
          <button
            key={name}
            type="button"
            onClick={() => setPart(name)}
            className={
              "h-11 rounded-md px-4 text-sm font-semibold " +
              (part === name ? "bg-copper text-copper-ink" : "border border-line bg-surface text-fg")
            }
          >
            {name === "Talk" ? `Talk · ${talk.length}` : `Q&A backup · ${SLIDES.length - talk.length}`}
          </button>
        ))}
      </div>

      <div className="flex gap-2 overflow-x-auto pb-1">
        {slides.map((s, i) => (
          <button
            key={s.id}
            type="button"
            onClick={() => setIndex(i)}
            className={
              "h-11 shrink-0 rounded-md px-3 text-sm font-semibold " +
              (i === index ? "bg-copper text-copper-ink" : "border border-line bg-surface text-muted")
            }
          >
            {s.id}
          </button>
        ))}
      </div>

      <SlideFace slide={slide} index={SLIDES.indexOf(slide)} />

      <div className="flex items-center justify-between gap-2">
        <button
          type="button"
          className="inline-flex h-11 items-center gap-1 rounded-md border border-line bg-surface px-3 text-sm font-semibold"
          onClick={() => setIndex((n) => Math.max(0, n - 1))}
          disabled={index === 0}
        >
          <ChevronLeft className="size-4" /> Prev
        </button>
        <p className="text-sm text-muted">
          {index + 1} / {slides.length}
          {slide.minutes ? ` · ${slide.minutes} min` : " · if asked"}
        </p>
        <button
          type="button"
          className="inline-flex h-11 items-center gap-1 rounded-md border border-line bg-surface px-3 text-sm font-semibold"
          onClick={() => setIndex((n) => Math.min(slides.length - 1, n + 1))}
          disabled={index === slides.length - 1}
        >
          Next <ChevronRight className="size-4" />
        </button>
      </div>

      <section className="rounded-md border border-line bg-surface p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="font-display text-lg font-semibold">Say this, not the bullets</h3>
          <CopyButton text={notesFor(slide)} label="Copy notes" />
        </div>
        <p className="mt-3 text-sm leading-relaxed">{slide.speak}</p>
        <p className="mt-3 text-sm leading-relaxed text-muted">
          <span className="font-semibold text-fg">Leave out. </span>
          {slide.avoid}
        </p>
        <p className="mt-3 text-sm leading-relaxed text-muted">
          <span className="font-semibold text-fg">How to draw it. </span>
          {slide.make}
        </p>
      </section>

      <CopyButton text={markdown} label="Copy the 6 talk slides" />
    </div>
  );
}
