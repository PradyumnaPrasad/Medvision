import type { AnalysisSummary } from "../types";

interface Props {
  analyses: AnalysisSummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
}

export function HistorySidebar({ analyses, activeId, onSelect }: Props) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-100 font-semibold text-slate-800 text-sm">
        History
      </div>
      {analyses.length === 0 ? (
        <p className="px-4 py-3 text-slate-400 text-sm">No analyses yet.</p>
      ) : (
        <ul className="divide-y divide-slate-50 max-h-96 overflow-y-auto">
          {analyses.map((a) => (
            <li key={a.id}>
              <button
                onClick={() => onSelect(a.id)}
                className={`w-full text-left px-4 py-2.5 hover:bg-sky-50 ${
                  a.id === activeId ? "bg-sky-50" : ""
                }`}
              >
                <div className="text-sm text-slate-800 truncate">
                  {a.top_differential ?? "No differential"}
                </div>
                <div className="text-xs text-slate-400 truncate">
                  {a.top_finding ?? "no findings"} ·{" "}
                  {new Date(a.created_at * 1000).toLocaleTimeString()}
                </div>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
