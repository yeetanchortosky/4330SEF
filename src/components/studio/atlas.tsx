import { useMemo, useState } from "react";
import { FINDINGS, THEMES, type Theme } from "@/lib/research/findings";

export function Atlas() {
  const [theme, setTheme] = useState<Theme | "All">("All");
  const [query, setQuery] = useState("");
  const items = useMemo(() => {
    const q = query.trim().toLowerCase();
    return FINDINGS.filter((item) => (theme === "All" || item.theme === theme) && (
      !q ||
      item.point.toLowerCase().includes(q) ||
      item.change.toLowerCase().includes(q) ||
      item.source.toLowerCase().includes(q)
    ));
  }, [theme, query]);

  return (
    <div className="flex flex-col gap-4">
      <div>
        <p className="text-xs font-semibold tracking-wide text-copper uppercase">{FINDINGS.length} implementation decisions</p>
        <h2 className="font-display mt-1 text-2xl leading-tight font-semibold">What the papers change in the build</h2>
        <p className="mt-2 text-sm leading-relaxed text-muted">
          Each card is a finding and the concrete choice it forces. The proposal’s references stay as written.
          Nothing here rewrites them.
        </p>
      </div>
      <input
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="Search a paper, a metric, a choice"
        className="h-12 rounded-md border border-line bg-surface px-3 text-base text-fg outline-none"
      />
      <div className="flex gap-2 overflow-x-auto pb-1">
        {(["All", ...THEMES] as const).map((name) => (
          <button
            key={name}
            type="button"
            onClick={() => setTheme(name)}
            className={
              "h-11 shrink-0 rounded-md px-3 text-sm font-semibold " +
              (theme === name ? "bg-copper text-copper-ink" : "border border-line bg-surface text-muted")
            }
          >
            {name}
          </button>
        ))}
      </div>
      <p className="text-sm text-muted">{items.length} shown</p>
      <ul className="flex flex-col gap-3">
        {items.map((item) => (
          <li key={item.id} className="rounded-md border border-line bg-surface p-4">
            <p className="text-xs font-semibold tracking-wide text-copper uppercase">
              {item.id} · {item.theme}
            </p>
            <p className="mt-2 text-sm leading-relaxed">{item.point}</p>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              <span className="font-semibold text-fg">Build it this way. </span>
              {item.change}
            </p>
            <p className="mt-2 text-xs text-faint">{item.source}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}
