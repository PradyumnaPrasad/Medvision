import type { RetrievalOut } from "../types";
import { GraphIcon } from "./icons";

interface Props {
  retrieval: RetrievalOut | null | undefined;
}

function Chip({ label, tone }: { label: string; tone: "normal" | "finding" }) {
  return (
    <span
      className={`px-2 py-0.5 rounded-full text-xs font-medium ring-1 ${
        tone === "normal"
          ? "bg-slate-50 text-slate-500 ring-slate-200 dark:bg-white/5 dark:text-slate-400 dark:ring-white/10"
          : "bg-sky-50 text-sky-700 ring-sky-100 dark:bg-sky-500/10 dark:text-sky-300 dark:ring-sky-500/20"
      }`}
    >
      {label}
    </span>
  );
}

/** Supporting context from the image <-> knowledge-graph alignment. It never changes the
 * diagnosis ranking (that is the deterministic KG reasoning); it shows which labelled
 * reference cases the image resembles and which KG nodes it sits closest to. */
export function RetrievalPanel({ retrieval }: Props) {
  if (!retrieval) {
    return (
      <div className="mv-card p-4 text-xs text-slate-400">
        Similar-case retrieval is not available for this analysis (the alignment index was not
        built when it was run).
      </div>
    );
  }
  return (
    <div className="mv-card overflow-hidden">
      <div className="mv-card-header">
        <GraphIcon className="w-4 h-4 text-indigo-500" />
        Similar reference cases
        <span className="text-slate-400 font-normal text-xs">(aligned image-graph space)</span>
      </div>
      <div className="px-4 py-3 space-y-2.5">
        {retrieval.similar_cases.map((c) => (
          <div key={c.case_id} className="flex items-center gap-3">
            <div className="flex-1 min-w-0 flex flex-wrap gap-1.5">
              {c.labels.map((label) => (
                <Chip
                  key={label}
                  label={label}
                  tone={label === "No Finding" ? "normal" : "finding"}
                />
              ))}
            </div>
            <span className="shrink-0 text-xs text-slate-400 font-mono" title={c.case_id}>
              {(c.similarity * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
      <div className="px-4 py-3 border-t border-slate-50 dark:border-white/5">
        <div className="text-[10px] uppercase tracking-wide text-slate-400 mb-1.5">
          Closest knowledge-graph nodes
        </div>
        <div className="flex flex-wrap gap-1.5">
          {retrieval.node_hints.map((h) => (
            <span
              key={h.node_id}
              className="px-2 py-0.5 rounded-lg text-xs bg-indigo-50 text-indigo-700 ring-1 ring-indigo-100 dark:bg-indigo-500/10 dark:text-indigo-300 dark:ring-indigo-500/20"
            >
              {h.name}
              <span className="text-indigo-400 font-mono"> {(h.score * 100).toFixed(0)}</span>
            </span>
          ))}
        </div>
        <p className="text-[11px] text-slate-400 mt-2.5 leading-snug">
          Supporting context only; it does not change the diagnosis ranking. Reference cases are
          a public NIH ChestX-ray14 sample.
        </p>
      </div>
    </div>
  );
}
