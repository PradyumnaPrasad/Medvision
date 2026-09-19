import { AlertIcon } from "./icons";

export function DisclaimerBanner({ text }: { text: string }) {
  return (
    <div className="w-full bg-amber-50 border-b border-amber-200 text-amber-900 dark:bg-amber-500/10 dark:border-amber-500/20 dark:text-amber-200 text-xs sm:text-sm px-4 py-2 text-center flex items-center justify-center gap-1.5">
      <AlertIcon className="w-3.5 h-3.5 shrink-0 text-amber-500" />
      <span>{text}</span>
    </div>
  );
}
