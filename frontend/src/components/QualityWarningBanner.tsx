import { AlertIcon } from "./icons";

export function QualityWarningBanner({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null;
  return (
    <div className="bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 text-sm text-amber-800 dark:bg-amber-500/10 dark:border-amber-500/20 dark:text-amber-300 mv-fade-in">
      <div className="font-medium mb-1 flex items-center gap-1.5">
        <AlertIcon className="w-4 h-4" />
        Image quality check flagged this upload
      </div>
      <ul className="list-disc list-inside space-y-0.5 pl-1">
        {warnings.map((w, i) => (
          <li key={i}>{w}</li>
        ))}
      </ul>
      <p className="mt-1.5 text-amber-700/80 text-xs">
        Findings below may be less reliable than usual — interpret with extra caution.
      </p>
    </div>
  );
}
