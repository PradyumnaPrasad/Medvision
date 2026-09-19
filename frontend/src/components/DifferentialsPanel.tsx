import { useState } from "react";
import type { DifferentialOut } from "../types";
import { ChevronDownIcon, GraphIcon } from "./icons";

function relationLabel(rel: string): string {
  return rel === "contradicts" ? "contradicts" : "suggestive of";
}

function EvidencePath({
  findingName,
  findingProb,
  relation,
  weight,
  contribution,
}: {
  findingName: string;
  findingProb: number;
  relation: string;
  weight: number;
  contribution: number;
}) {
  const positive = contribution >= 0;
  return (
    <div className="flex items-center gap-2 text-xs flex-wrap">
      <span className="shrink-0 px-2 py-1 rounded-lg bg-sky-50 text-sky-700 ring-1 ring-sky-100 font-medium whitespace-nowrap dark:bg-sky-500/10 dark:text-sky-300 dark:ring-sky-500/20">
        {findingName}
        <span className="text-sky-400 font-normal"> {(findingProb * 100).toFixed(0)}%</span>
      </span>

      <span
        className={`shrink-0 h-px w-4 sm:w-8 ${relation === "contradicts" ? "bg-rose-300 dark:bg-rose-500/40" : "bg-indigo-300 dark:bg-indigo-500/40"}`}
      />

      <span className="shrink-0 text-slate-400 whitespace-nowrap">
        {relationLabel(relation)}{" "}
        <span className="text-slate-300 dark:text-slate-600">(w={weight.toFixed(2)})</span>
      </span>

      <span
        className={`shrink-0 font-mono ml-auto pl-2 ${positive ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}
      >
        {positive ? "+" : ""}
        {contribution.toFixed(3)}
      </span>
    </div>
  );
}

export function DifferentialsPanel({ differentials }: { differentials: DifferentialOut[] }) {
  const [expanded, setExpanded] = useState<string | null>(differentials[0]?.node_id ?? null);

  if (differentials.length === 0) {
    return (
      <div className="mv-card p-4 text-slate-500 text-sm">
        No differential diagnosis reached the activation threshold for this image.
      </div>
    );
  }

  return (
    <div className="mv-card overflow-hidden">
      <div className="mv-card-header">
        <GraphIcon className="w-4 h-4 text-indigo-500" />
        Differential diagnoses
        <span className="text-slate-400 font-normal text-xs">(knowledge-graph reasoning)</span>
      </div>
      <div className="divide-y divide-slate-50 dark:divide-white/5">
        {differentials.map((d, i) => {
          const isOpen = expanded === d.node_id;
          return (
            <div key={d.node_id}>
              <button
                onClick={() => setExpanded(isOpen ? null : d.node_id)}
                className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-slate-50/70 dark:hover:bg-white/5 transition-colors"
              >
                <span
                  className={`text-xs w-5 h-5 rounded-full flex items-center justify-center font-semibold shrink-0 ${
                    i === 0
                      ? "bg-indigo-100 text-indigo-700 dark:bg-indigo-500/20 dark:text-indigo-300"
                      : "bg-slate-100 text-slate-500 dark:bg-white/10 dark:text-slate-400"
                  }`}
                >
                  {i + 1}
                </span>
                <span className="flex-1 text-slate-800 dark:text-slate-100 font-medium">
                  {d.name}
                </span>
                <div className="w-24 h-1.5 bg-slate-100 dark:bg-white/10 rounded-full overflow-hidden hidden sm:block">
                  <div
                    className="h-full bg-indigo-500 rounded-full"
                    style={{ width: `${Math.round(d.score * 100)}%` }}
                  />
                </div>
                <span className="text-slate-500 text-xs w-10 text-right tabular-nums">
                  {(d.score * 100).toFixed(0)}%
                </span>
                <ChevronDownIcon
                  className={`w-3.5 h-3.5 text-slate-400 transition-transform ${isOpen ? "rotate-180" : ""}`}
                />
              </button>
              {isOpen && (
                <div className="px-4 pb-3.5 pl-8 space-y-2 mv-fade-in">
                  {d.paths.map((p, idx) => (
                    <EvidencePath
                      key={idx}
                      findingName={p.finding_name}
                      findingProb={p.finding_prob}
                      relation={p.relation}
                      weight={p.weight}
                      contribution={p.contribution}
                    />
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
