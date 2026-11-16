import { reportPdfUrl } from "../api";
import type { AnalysisResult } from "../types";
import { CheckCircleIcon, DownloadIcon, SparkleIcon } from "./icons";

function groundingColor(score: number): string {
  if (score >= 0.9) return "text-emerald-500 dark:text-emerald-400";
  if (score >= 0.7) return "text-amber-500 dark:text-amber-400";
  return "text-rose-500 dark:text-rose-400";
}

/** A single at-a-glance summary shown above the detailed panels — the "what do I
 * need to know in 3 seconds" view, before someone reads the full findings table,
 * evidence graph, and report below. */
export function SummaryCard({ result }: { result: AnalysisResult }) {
  const shown = result.findings.filter((f) => f.status !== "absent");
  const top = result.differentials[0];

  return (
    <div className="mv-card mv-fade-in p-4 sm:p-5">
      <div className="flex flex-wrap items-start gap-x-2 gap-y-2 mb-3">
        <SparkleIcon className="w-4 h-4 text-indigo-400 mt-0.5 shrink-0" />
        <p className="flex-1 min-w-[14rem] text-sm text-slate-700 dark:text-slate-200 leading-relaxed">
          {top ? (
            <>
              Findings are most consistent with{" "}
              <span className="font-semibold text-slate-900 dark:text-white">{top.name}</span>{" "}
              ({(top.score * 100).toFixed(0)}% score), based on{" "}
              <span className="font-medium">{shown.length}</span> finding
              {shown.length === 1 ? "" : "s"} above the reporting threshold.
            </>
          ) : (
            "No differential diagnosis reached the activation threshold for this image."
          )}
        </p>
        <a
          href={reportPdfUrl(result.id)}
          download
          title="Download this analysis as a PDF report"
          className="shrink-0 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-indigo-700 bg-indigo-50 ring-1 ring-indigo-100 hover:bg-indigo-100 dark:text-indigo-300 dark:bg-indigo-500/10 dark:ring-indigo-500/20 dark:hover:bg-indigo-500/20 transition-colors"
        >
          <DownloadIcon className="w-3.5 h-3.5" />
          PDF report
        </a>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-3 border-t border-slate-100 dark:border-white/10">
        <Stat label="Top diagnosis" value={top?.name ?? "—"} small />
        <Stat
          label="Confidence"
          value={top ? `${(top.score * 100).toFixed(0)}%` : "—"}
          valueClassName="text-indigo-600 dark:text-indigo-400"
        />
        <Stat
          label="Grounding"
          value={`${(result.grounding.score * 100).toFixed(0)}%`}
          valueClassName={groundingColor(result.grounding.score)}
          icon={<CheckCircleIcon className="w-3.5 h-3.5" />}
        />
        <Stat label="Report source" value={result.report_source === "llm" ? "LLM" : "Template"} />
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  valueClassName = "text-slate-800 dark:text-slate-100",
  small = false,
  icon,
}: {
  label: string;
  value: string;
  valueClassName?: string;
  small?: boolean;
  icon?: React.ReactNode;
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide text-slate-400 mb-0.5">{label}</div>
      <div
        className={`font-semibold flex items-center gap-1 ${small ? "text-sm" : "text-lg"} ${valueClassName}`}
      >
        {icon}
        <span className="truncate">{value}</span>
      </div>
    </div>
  );
}
