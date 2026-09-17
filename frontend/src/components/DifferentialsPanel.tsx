import { useState } from "react";
import type { DifferentialOut } from "../types";

function relationLabel(rel: string): string {
  return rel === "contradicts" ? "contradicts" : "suggestive of";
}

export function DifferentialsPanel({ differentials }: { differentials: DifferentialOut[] }) {
  const [expanded, setExpanded] = useState<string | null>(differentials[0]?.node_id ?? null);

  if (differentials.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-4 text-slate-500 text-sm">
        No differential diagnosis reached the activation threshold for this image.
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-100 font-semibold text-slate-800">
        Differential diagnoses{" "}
        <span className="text-slate-400 font-normal text-xs">(knowledge-graph reasoning)</span>
      </div>
      <div className="divide-y divide-slate-50">
        {differentials.map((d, i) => {
          const isOpen = expanded === d.node_id;
          return (
            <div key={d.node_id}>
              <button
                onClick={() => setExpanded(isOpen ? null : d.node_id)}
                className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-slate-50"
              >
                <span className="text-slate-400 text-xs w-5">#{i + 1}</span>
                <span className="flex-1 text-slate-800 font-medium">{d.name}</span>
                <div className="w-28 h-2 bg-slate-100 rounded overflow-hidden">
                  <div
                    className="h-full bg-indigo-500"
                    style={{ width: `${Math.round(d.score * 100)}%` }}
                  />
                </div>
                <span className="text-slate-500 text-xs w-10 text-right">
                  {(d.score * 100).toFixed(0)}%
                </span>
                <span className="text-slate-400 text-xs">{isOpen ? "▲" : "▼"}</span>
              </button>
              {isOpen && (
                <div className="px-4 pb-3 pl-12 space-y-1.5">
                  {d.paths.map((p, idx) => (
                    <div
                      key={idx}
                      className="text-xs text-slate-600 font-mono bg-slate-50 rounded px-2 py-1.5"
                    >
                      <span className="text-slate-800">{p.finding_name}</span>
                      <span className="text-slate-400"> ({(p.finding_prob * 100).toFixed(0)}%)</span>
                      <span className="text-indigo-500">
                        {" "}
                        —{relationLabel(p.relation)} (w={p.weight.toFixed(2)})→{" "}
                      </span>
                      <span className="text-slate-800">{d.name}</span>
                      <span
                        className={p.contribution >= 0 ? "text-emerald-600" : "text-rose-600"}
                      >
                        {"  "}
                        {p.contribution >= 0 ? "+" : ""}
                        {p.contribution.toFixed(3)}
                      </span>
                    </div>
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
