import type { GroundingResult, ReportSource } from "../types";
import { AlertIcon, CheckCircleIcon } from "./icons";

interface Props {
  source: ReportSource;
  grounding: GroundingResult;
}

function scoreStyle(score: number): string {
  if (score >= 0.9) return "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200";
  if (score >= 0.7) return "bg-amber-50 text-amber-700 ring-1 ring-amber-200";
  return "bg-rose-50 text-rose-700 ring-1 ring-rose-200";
}

export function GroundingBadge({ source, grounding }: Props) {
  const issues = [...grounding.unsupported, ...grounding.contradictions];
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-600 ring-1 ring-slate-200/70">
          {source === "llm" ? "LLM-narrated" : "Template (LLM unavailable)"}
        </span>
        <span
          className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium ${scoreStyle(grounding.score)}`}
          title="Fraction of report statements whose citations and claims checked out against the evidence"
        >
          <CheckCircleIcon className="w-3 h-3" />
          Grounding {(grounding.score * 100).toFixed(0)}%
        </span>
      </div>
      {issues.length > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl px-3 py-2.5 text-xs text-amber-800 space-y-1">
          <div className="font-medium flex items-center gap-1.5">
            <AlertIcon className="w-3.5 h-3.5" />
            The verifier flagged issues in this draft
          </div>
          {issues.map((msg, i) => (
            <div key={i} className="pl-5">
              • {msg}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
