import { useState } from "react";
import { submitFeedback } from "../api";
import type { FindingOut } from "../types";

interface Props {
  analysisId: string;
  findings: FindingOut[];
  selectedNodeId: string | null;
  onSelect: (nodeId: string) => void;
}

const STATUS_STYLES: Record<string, string> = {
  present: "bg-emerald-100 text-emerald-800",
  possible: "bg-amber-100 text-amber-800",
  absent: "bg-slate-100 text-slate-500",
};

function formatRegion(region: string | null): string {
  if (!region) return "—";
  // A_RIGHT_LOWER_ZONE -> Right Lower Zone
  return region
    .replace(/^A_/, "")
    .split("_")
    .map((w) => w[0] + w.slice(1).toLowerCase())
    .join(" ");
}

export function FindingsTable({ analysisId, findings, selectedNodeId, onSelect }: Props) {
  const [feedbackGiven, setFeedbackGiven] = useState<Record<string, "agree" | "disagree">>({});

  const giveFeedback = async (nodeId: string, verdict: "agree" | "disagree") => {
    setFeedbackGiven((prev) => ({ ...prev, [nodeId]: verdict }));
    try {
      await submitFeedback(analysisId, nodeId, verdict);
    } catch {
      // best-effort: feedback is a nice-to-have, don't block the UI on a failed POST
    }
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-100 font-semibold text-slate-800">
        Findings
      </div>
      <table className="w-full text-sm">
        <thead className="text-left text-slate-500">
          <tr>
            <th className="px-4 py-2 font-medium">Finding</th>
            <th className="px-4 py-2 font-medium">Probability</th>
            <th className="px-4 py-2 font-medium">Status</th>
            <th className="px-4 py-2 font-medium">Region</th>
            <th className="px-4 py-2 font-medium">Feedback</th>
          </tr>
        </thead>
        <tbody>
          {findings.map((f) => {
            const clickable = f.status !== "absent";
            const isSelected = f.node_id === selectedNodeId;
            return (
              <tr
                key={f.node_id}
                onClick={() => clickable && onSelect(f.node_id)}
                className={`border-t border-slate-50 ${clickable ? "cursor-pointer hover:bg-sky-50" : ""} ${
                  isSelected ? "bg-sky-50" : ""
                }`}
              >
                <td className="px-4 py-2 text-slate-800">{f.name}</td>
                <td className="px-4 py-2 w-40">
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-2 bg-slate-100 rounded overflow-hidden">
                      <div
                        className="h-full bg-sky-500"
                        style={{ width: `${Math.round(f.prob * 100)}%` }}
                      />
                    </div>
                    <span className="text-slate-500 text-xs w-10 text-right">
                      {(f.prob * 100).toFixed(0)}%
                    </span>
                  </div>
                </td>
                <td className="px-4 py-2">
                  <span
                    className={`px-2 py-0.5 rounded-full text-xs font-medium ${STATUS_STYLES[f.status]}`}
                  >
                    {f.status}
                  </span>
                </td>
                <td className="px-4 py-2 text-slate-500">{formatRegion(f.region)}</td>
                <td className="px-4 py-2" onClick={(e) => e.stopPropagation()}>
                  {clickable ? (
                    feedbackGiven[f.node_id] ? (
                      <span className="text-slate-400 text-xs">
                        {feedbackGiven[f.node_id] === "agree" ? "👍 saved" : "👎 saved"}
                      </span>
                    ) : (
                      <div className="flex gap-1">
                        <button
                          onClick={() => giveFeedback(f.node_id, "agree")}
                          className="hover:bg-emerald-50 rounded px-1.5 py-0.5"
                          title="Agree with this finding"
                        >
                          👍
                        </button>
                        <button
                          onClick={() => giveFeedback(f.node_id, "disagree")}
                          className="hover:bg-rose-50 rounded px-1.5 py-0.5"
                          title="Disagree with this finding"
                        >
                          👎
                        </button>
                      </div>
                    )
                  ) : (
                    <span className="text-slate-300 text-xs">—</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
