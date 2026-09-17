import { CitedText } from "./CitedText";
import { GroundingBadge } from "./GroundingBadge";
import type { AnalysisResult } from "../types";

type Props = Pick<
  AnalysisResult,
  "report" | "report_source" | "grounding" | "evidence" | "timings_ms"
>;

export function ReportPanel({ report, report_source, grounding, evidence, timings_ms }: Props) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div className="font-semibold text-slate-800">Draft report</div>
      </div>

      <GroundingBadge source={report_source} grounding={grounding} />

      <section>
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">
          Findings
        </div>
        <div className="space-y-1 text-sm text-slate-700 leading-relaxed">
          {report.findings.length === 0 && (
            <p className="text-slate-400">No findings met the reporting threshold.</p>
          )}
          {report.findings.map((item, i) => (
            <p key={i}>
              <CitedText text={item.text} evidenceIds={item.evidence_ids} evidence={evidence} />
            </p>
          ))}
        </div>
      </section>

      <section>
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">
          Impression
        </div>
        <div className="space-y-1 text-sm text-slate-700 leading-relaxed">
          {report.impression.length === 0 && (
            <p className="text-slate-400">No impression could be generated.</p>
          )}
          {report.impression.map((item, i) => (
            <p key={i}>
              <CitedText text={item.text} evidenceIds={item.evidence_ids} evidence={evidence} />
            </p>
          ))}
        </div>
      </section>

      {report.differentials.length > 0 && (
        <section>
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">
            Differential diagnoses
          </div>
          <div className="space-y-2">
            {report.differentials.map((item, i) => (
              <div key={i} className="text-sm">
                <span className="font-medium text-slate-800">{item.diagnosis}: </span>
                <span className="text-slate-600">
                  <CitedText
                    text={item.rationale}
                    evidenceIds={item.evidence_ids}
                    evidence={evidence}
                  />
                </span>
              </div>
            ))}
          </div>
        </section>
      )}

      {report.limitations.length > 0 && (
        <section>
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">
            Limitations
          </div>
          <ul className="list-disc list-inside text-sm text-slate-600 space-y-0.5">
            {report.limitations.map((lim, i) => (
              <li key={i}>{lim}</li>
            ))}
          </ul>
        </section>
      )}

      <section>
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">
          Recommendation
        </div>
        <p className="text-sm text-slate-700">{report.recommendation}</p>
      </section>

      <div className="pt-3 border-t border-slate-100 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-400">
        {Object.entries(timings_ms).map(([key, value]) => (
          <span key={key}>
            {key.replace("_ms", "")}: {value.toFixed(0)}ms
          </span>
        ))}
      </div>
    </div>
  );
}
