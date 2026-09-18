export function QualityWarningBanner({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null;
  return (
    <div className="bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 text-sm text-amber-800">
      <div className="font-medium mb-1">⚠️ Image quality check flagged this upload</div>
      <ul className="list-disc list-inside space-y-0.5">
        {warnings.map((w, i) => (
          <li key={i}>{w}</li>
        ))}
      </ul>
      <p className="mt-1 text-amber-700/80 text-xs">
        Findings below may be less reliable than usual — interpret with extra caution.
      </p>
    </div>
  );
}
