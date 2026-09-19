import { CitedText } from "./CitedText";
import { GroundingBadge } from "./GroundingBadge";
import { ClockIcon, DocumentIcon } from "./icons";
import type { AnalysisResult } from "../types";

type Props = Pick<
  AnalysisResult,
  "report" | "report_source" | "grounding" | "evidence" | "timings_ms"
>;

function Section({
  title,
  accent,
  children,
}: {
  title: string;
  accent: string;
  children: React.ReactNode;
}) {
  return (
    <section className="relative pl-3.5">
      <span className={`absolute left-0 top-0.5 bottom-0.5 w-0.5 rounded-full ${accent}`} />
      <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">
        {title}
      </div>
      {children}
    </section>
  );
}

export function ReportPanel({ report, report_source, grounding, evidence, timings_ms }: Props) {
  return (
    <div className="mv-card p-4 space-y-4">
      <div className="flex items-center gap-2">
        <DocumentIcon className="w-4 h-4 text-slate-400" />
        <div className="font-semibold text-slate-800">Draft report</div>
      </div>

      <GroundingBadge source={report_source} grounding={grounding} />

      <Section title="Findings" accent="bg-sky-300">
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
      </Section>

      <Section title="Impression" accent="bg-indigo-300">
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
      </Section>

      {report.differentials.length > 0 && (
        <Section title="Differential diagnoses" accent="bg-purple-300">
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
        </Section>
      )}

      {report.limitations.length > 0 && (
        <Section title="Limitations" accent="bg-slate-300">
          <ul className="list-disc list-inside text-sm text-slate-600 space-y-0.5">
            {report.limitations.map((lim, i) => (
              <li key={i}>{lim}</li>
            ))}
          </ul>
        </Section>
      )}

      <Section title="Recommendation" accent="bg-emerald-300">
        <p className="text-sm text-slate-700">{report.recommendation}</p>
      </Section>

      <div className="pt-3 border-t border-slate-100 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-400">
        <ClockIcon className="w-3 h-3" />
        {Object.entries(timings_ms).map(([key, value]) => (
          <span key={key} className="tabular-nums">
            {key.replace("_ms", "")}: {value.toFixed(0)}ms
          </span>
        ))}
      </div>
    </div>
  );
}
