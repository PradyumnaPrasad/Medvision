import { useEffect, useState } from "react";

type Theme = "light" | "dark";
const STORAGE_KEY = "medvision-theme";

function getInitialTheme(): Theme {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "light" || stored === "dark") return stored;
  } catch {
    // localStorage can throw in private-browsing/blocked-storage contexts — fall
    // through to the system preference rather than crashing the app over a theme.
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/** Manual dark/light toggle, persisted to localStorage, applied via a `.dark`
 * class on <html> (see index.css's @custom-variant dark rule). Falls back to the
 * system preference the first time, then remembers whatever the user picked. */
export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      // best-effort persistence; the toggle still works for this page view
    }
  }, [theme]);

  const toggle = () => setTheme((t) => (t === "dark" ? "light" : "dark"));
  return [theme, toggle];
}
