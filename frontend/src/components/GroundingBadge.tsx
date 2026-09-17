import type { GroundingResult, ReportSource } from "../types";

interface Props {
  source: ReportSource;
  grounding: GroundingResult;
}

function scoreStyle(score: number): string {
  if (score >= 0.9) return "bg-emerald-100 text-emerald-800";
  if (score >= 0.7) return "bg-amber-100 text-amber-800";
  return "bg-rose-100 text-rose-800";
}

export function GroundingBadge({ source, grounding }: Props) {
  const issues = [...grounding.unsupported, ...grounding.contradictions];
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-slate-100 text-slate-600">
          {source === "llm" ? "LLM-narrated" : "Template (LLM unavailable)"}
        </span>
        <span
          className={`px-2 py-0.5 rounded-full text-xs font-medium ${scoreStyle(grounding.score)}`}
          title="Fraction of report statements whose citations and claims checked out against the evidence"
        >
          Grounding: {(grounding.score * 100).toFixed(0)}%
        </span>
      </div>
      {issues.length > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-xs text-amber-800 space-y-1">
          <div className="font-medium">⚠️ The verifier flagged issues in this draft:</div>
          {issues.map((msg, i) => (
            <div key={i}>• {msg}</div>
          ))}
        </div>
      )}
    </div>
  );
}
