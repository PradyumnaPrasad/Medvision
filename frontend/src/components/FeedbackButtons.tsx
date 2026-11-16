import type { FeedbackVerdict } from "../types";
import { ThumbsDownIcon, ThumbsUpIcon } from "./icons";

interface Props {
  verdict: FeedbackVerdict | undefined;
  onChange: (verdict: FeedbackVerdict) => void;
  subject: string;
}

const BASE = "rounded-md p-1.5 transition-colors";

/** Agree/disagree control. The chosen verdict stays highlighted and can be changed. */
export function FeedbackButtons({ verdict, onChange, subject }: Props) {
  return (
    <div className="flex gap-1" role="group" aria-label={`Feedback on ${subject}`}>
      <button
        type="button"
        onClick={() => onChange("agree")}
        aria-pressed={verdict === "agree"}
        title={`Agree with ${subject}`}
        className={`${BASE} ${
          verdict === "agree"
            ? "bg-emerald-50 text-emerald-600 ring-1 ring-emerald-200 dark:bg-emerald-500/15 dark:text-emerald-400 dark:ring-emerald-500/30"
            : "text-slate-400 hover:text-emerald-600 hover:bg-emerald-50 dark:hover:bg-emerald-500/10 dark:hover:text-emerald-400"
        }`}
      >
        <ThumbsUpIcon />
      </button>
      <button
        type="button"
        onClick={() => onChange("disagree")}
        aria-pressed={verdict === "disagree"}
        title={`Disagree with ${subject}`}
        className={`${BASE} ${
          verdict === "disagree"
            ? "bg-rose-50 text-rose-600 ring-1 ring-rose-200 dark:bg-rose-500/15 dark:text-rose-400 dark:ring-rose-500/30"
            : "text-slate-400 hover:text-rose-600 hover:bg-rose-50 dark:hover:bg-rose-500/10 dark:hover:text-rose-400"
        }`}
      >
        <ThumbsDownIcon />
      </button>
    </div>
  );
}
