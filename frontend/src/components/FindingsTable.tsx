import { useState } from "react";
import { submitFeedback } from "../api";
import type { FindingOut } from "../types";
import { ChevronDownIcon, CheckCircleIcon, ThumbsDownIcon, ThumbsUpIcon } from "./icons";

interface Props {
  analysisId: string;
  findings: FindingOut[];
  selectedNodeId: string | null;
  onSelect: (nodeId: string) => void;
}

const STATUS_STYLES: Record<string, string> = {
  present: "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200",
  possible: "bg-amber-50 text-amber-700 ring-1 ring-amber-200",
  absent: "bg-slate-50 text-slate-400 ring-1 ring-slate-200",
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
  const [showAll, setShowAll] = useState(false);

  const giveFeedback = async (nodeId: string, verdict: "agree" | "disagree") => {
    setFeedbackGiven((prev) => ({ ...prev, [nodeId]: verdict }));
    try {
      await submitFeedback(analysisId, nodeId, verdict);
    } catch {
      // best-effort: feedback is a nice-to-have, don't block the UI on a failed POST
    }
  };

  const shown = findings.filter((f) => f.status !== "absent");
  const hidden = findings.filter((f) => f.status === "absent");
  const rows = showAll ? findings : shown;
  const hasCalibrated = findings.some((f) => Math.abs(f.prob - f.raw_prob) >= 0.01);

  return (
    <div className="mv-card overflow-hidden">
      <div className="mv-card-header justify-between">
        <span>Findings</span>
        <span className="text-xs font-normal text-slate-400">
          {shown.length} of {findings.length} above threshold
        </span>
      </div>
      <table className="w-full text-sm">
        <thead className="text-left text-slate-400">
          <tr>
            <th className="px-4 pt-3 pb-2 font-medium text-xs uppercase tracking-wide">Finding</th>
            <th className="px-4 pt-3 pb-2 font-medium text-xs uppercase tracking-wide">
              Probability
            </th>
            <th className="px-4 pt-3 pb-2 font-medium text-xs uppercase tracking-wide">Status</th>
            <th className="px-4 pt-3 pb-2 font-medium text-xs uppercase tracking-wide">Region</th>
            <th className="px-4 pt-3 pb-2 font-medium text-xs uppercase tracking-wide">
              Feedback
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((f) => {
            const clickable = f.status !== "absent";
            const isSelected = f.node_id === selectedNodeId;
            return (
              <tr
                key={f.node_id}
                onClick={() => clickable && onSelect(f.node_id)}
                className={`border-t border-l-2 border-slate-50 transition-colors ${
                  clickable ? "cursor-pointer hover:bg-sky-50/70" : "opacity-60"
                } ${isSelected ? "bg-sky-50 border-l-sky-500" : "border-l-transparent"}`}
              >
                <td className="px-4 py-2.5 text-slate-800 font-medium">{f.name}</td>
                <td className="px-4 py-2.5 w-40">
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all ${
                          f.status === "present"
                            ? "bg-emerald-500"
                            : f.status === "possible"
                              ? "bg-amber-400"
                              : "bg-slate-300"
                        }`}
                        style={{ width: `${Math.round(f.prob * 100)}%` }}
                      />
                    </div>
                    <span
                      className="text-slate-500 text-xs w-10 text-right tabular-nums"
                      title={
                        Math.abs(f.prob - f.raw_prob) >= 0.01
                          ? `Calibrated from raw model output of ${(f.raw_prob * 100).toFixed(0)}%`
                          : "Raw model output (no calibration fitted for this finding yet)"
                      }
                    >
                      {(f.prob * 100).toFixed(0)}%
                      {Math.abs(f.prob - f.raw_prob) >= 0.01 && (
                        <sup className="text-indigo-400">*</sup>
                      )}
                    </span>
                  </div>
                </td>
                <td className="px-4 py-2.5">
                  <span
                    className={`px-2 py-0.5 rounded-full text-xs font-medium capitalize ${STATUS_STYLES[f.status]}`}
                  >
                    {f.status}
                  </span>
                </td>
                <td className="px-4 py-2.5 text-slate-500 text-xs">{formatRegion(f.region)}</td>
                <td className="px-4 py-2.5" onClick={(e) => e.stopPropagation()}>
                  {clickable ? (
                    feedbackGiven[f.node_id] ? (
                      <span className="text-emerald-500 text-xs flex items-center gap-1">
                        <CheckCircleIcon className="w-3.5 h-3.5" /> saved
                      </span>
                    ) : (
                      <div className="flex gap-1">
                        <button
                          onClick={() => giveFeedback(f.node_id, "agree")}
                          className="text-slate-400 hover:text-emerald-600 hover:bg-emerald-50 rounded-md p-1.5 transition-colors"
                          title="Agree with this finding"
                        >
                          <ThumbsUpIcon />
                        </button>
                        <button
                          onClick={() => giveFeedback(f.node_id, "disagree")}
                          className="text-slate-400 hover:text-rose-600 hover:bg-rose-50 rounded-md p-1.5 transition-colors"
                          title="Disagree with this finding"
                        >
                          <ThumbsDownIcon />
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
      {hidden.length > 0 && (
        <button
          onClick={() => setShowAll((v) => !v)}
          className="w-full flex items-center justify-center gap-1.5 px-4 py-2.5 text-xs font-medium text-slate-500 hover:bg-slate-50 border-t border-slate-100 transition-colors"
        >
          <ChevronDownIcon
            className={`w-3.5 h-3.5 transition-transform ${showAll ? "rotate-180" : ""}`}
          />
          {showAll ? "Hide findings below threshold" : `Show ${hidden.length} more (below threshold)`}
        </button>
      )}
      {hasCalibrated && (
        <div className="px-4 py-2 text-xs text-slate-400 border-t border-slate-50">
          <sup className="text-indigo-400">*</sup> calibrated against real labeled data — hover a
          percentage to see the raw model output
        </div>
      )}
    </div>
  );
}
