import type { AnalysisSummary } from "../types";
import { ClockIcon } from "./icons";

interface Props {
  analyses: AnalysisSummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
}

export function HistorySidebar({ analyses, activeId, onSelect }: Props) {
  return (
    <div className="mv-card overflow-hidden">
      <div className="mv-card-header">
        <ClockIcon className="w-3.5 h-3.5 text-slate-400" />
        History
      </div>
      {analyses.length === 0 ? (
        <p className="px-4 py-4 text-slate-400 text-sm text-center">No analyses yet.</p>
      ) : (
        <ul className="divide-y divide-slate-50 dark:divide-white/5 max-h-96 overflow-y-auto">
          {analyses.map((a) => {
            const isActive = a.id === activeId;
            return (
              <li key={a.id}>
                <button
                  onClick={() => onSelect(a.id)}
                  className={`w-full text-left px-4 py-2.5 border-l-2 transition-colors ${
                    isActive
                      ? "bg-sky-50 border-l-sky-500 dark:bg-sky-500/10"
                      : "border-l-transparent hover:bg-slate-50 dark:hover:bg-white/5"
                  }`}
                >
                  <div className="text-sm text-slate-800 dark:text-slate-100 font-medium truncate">
                    {a.top_differential ?? "No differential"}
                  </div>
                  <div className="text-xs text-slate-400 truncate">
                    {a.top_finding ?? "no findings"} ·{" "}
                    {new Date(a.created_at * 1000).toLocaleTimeString()}
                  </div>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
