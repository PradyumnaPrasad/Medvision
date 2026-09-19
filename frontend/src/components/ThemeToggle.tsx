import { MoonIcon, SunIcon } from "./icons";

export function ThemeToggle({
  theme,
  onToggle,
}: {
  theme: "light" | "dark";
  onToggle: () => void;
}) {
  return (
    <button
      onClick={onToggle}
      title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
      className="w-9 h-9 rounded-full flex items-center justify-center text-slate-500 hover:bg-slate-900/5 dark:text-slate-300 dark:hover:bg-white/10 transition-colors shrink-0"
    >
      {theme === "dark" ? <SunIcon className="w-4 h-4" /> : <MoonIcon className="w-4 h-4" />}
    </button>
  );
}
