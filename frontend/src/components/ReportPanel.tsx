interface Props {
  reportText: string;
  timingsMs: Record<string, number>;
}

export function ReportPanel({ reportText, timingsMs }: Props) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4">
      <div className="font-semibold text-slate-800 mb-2">Draft report</div>
      <pre className="whitespace-pre-wrap font-sans text-sm text-slate-700 leading-relaxed">
        {reportText}
      </pre>
      <div className="mt-4 pt-3 border-t border-slate-100 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-400">
        {Object.entries(timingsMs).map(([key, value]) => (
          <span key={key}>
            {key.replace("_ms", "")}: {value.toFixed(0)}ms
          </span>
        ))}
      </div>
    </div>
  );
}
