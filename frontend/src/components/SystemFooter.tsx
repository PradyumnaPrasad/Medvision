import type { HealthOut } from "../types";

function StatusDot({ ok }: { ok: boolean }) {
  return (
    <span
      className={`w-1.5 h-1.5 rounded-full ${ok ? "bg-emerald-500" : "bg-rose-500"} ${ok ? "animate-pulse" : ""}`}
    />
  );
}

export function SystemFooter({ health }: { health: HealthOut | null }) {
  return (
    <footer className="max-w-7xl mx-auto px-4 py-6 mt-6 border-t border-slate-200/80 dark:border-white/10 w-full">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-slate-400">
        {health ? (
          <>
            <span className="flex items-center gap-1.5">
              <StatusDot ok /> backend connected
            </span>
            <span className="text-slate-300 dark:text-slate-600">·</span>
            <span>
              Inference device:{" "}
              <span className="font-mono text-slate-500 dark:text-slate-400">{health.device}</span>
            </span>
            <span className="text-slate-300 dark:text-slate-600">·</span>
            <span>
              Knowledge graph: {health.kg_nodes} nodes, {health.kg_edges} edges
            </span>
            <span className="text-slate-300 dark:text-slate-600">·</span>
            <span className="flex items-center gap-1.5">
              <StatusDot ok={!!health.retrieval_ready} /> case retrieval{" "}
              {health.retrieval_ready ? "ready" : "not built"}
            </span>
          </>
        ) : (
          <span className="flex items-center gap-1.5 text-rose-500">
            <StatusDot ok={false} /> backend unreachable
          </span>
        )}
      </div>
      <p className="text-xs text-slate-300 dark:text-slate-600 mt-2">
        MEDVISION is a college research prototype. It is not a certified medical device and must
        never be used for real clinical decisions.
      </p>
    </footer>
  );
}
