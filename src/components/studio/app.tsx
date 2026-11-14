import { useState } from "react";
import { BookOpen, FlaskConical, ListChecks, ScrollText } from "lucide-react";
import { Bench } from "./bench";
import { Deck } from "./deck";
import { Manual } from "./manual";
import { Study } from "./study";

const VIEWS = [
  { id: "study", label: "Study", icon: ListChecks },
  { id: "lab", label: "Lab", icon: FlaskConical },
  { id: "deck", label: "Slides", icon: BookOpen },
  { id: "manual", label: "Manual", icon: ScrollText },
] as const;

type ViewId = (typeof VIEWS)[number]["id"];

export function Studio() {
  const [view, setView] = useState<ViewId>("study");
  return (
    <div className="min-h-screen bg-bg text-fg">
      <header className="border-b border-line px-4 py-4">
        <p className="text-xs font-semibold tracking-wide text-copper uppercase">COMP 4330SEF · Group 5 · P08</p>
        <h1 className="font-display text-3xl leading-none font-semibold">Orderline</h1>
        <p className="mt-2 max-w-xl text-sm leading-relaxed text-muted">
          The proposal’s LightGCN experiment: baselines, one-factor ablation, the five-interaction slice, then one
          small change.
        </p>
      </header>
      <div className="mx-auto flex w-full max-w-3xl flex-col gap-4 px-4 pt-4 pb-28">
        {view === "study" ? <Study /> : null}
        {view === "lab" ? <Bench /> : null}
        {view === "deck" ? <Deck /> : null}
        {view === "manual" ? <Manual /> : null}
      </div>
      <nav className="fixed inset-x-0 bottom-0 z-10 border-t border-line bg-bg">
        <ul className="mx-auto grid max-w-3xl grid-cols-4">
          {VIEWS.map((item) => {
            const Icon = item.icon;
            const on = view === item.id;
            return (
              <li key={item.id}>
                <button
                  type="button"
                  onClick={() => setView(item.id)}
                  className={"flex h-16 w-full flex-col items-center justify-center gap-1 text-xs font-semibold " + (on ? "text-copper" : "text-muted")}
                >
                  <Icon className="size-5" />
                  {item.label}
                </button>
              </li>
            );
          })}
        </ul>
      </nav>
    </div>
  );
}
