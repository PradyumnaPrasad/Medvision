import type { HealthOut } from "../types";

export function SystemFooter({ health }: { health: HealthOut | null }) {
  return (
    <footer className="max-w-7xl mx-auto px-4 py-6 mt-6 border-t border-slate-200">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-400">
        {health ? (
          <>
            <span>
              Inference device: <span className="font-mono">{health.device}</span>
            </span>
            <span>·</span>
            <span>
              Knowledge graph: {health.kg_nodes} nodes, {health.kg_edges} edges
            </span>
            <span>·</span>
            <span className="text-emerald-600">● backend connected</span>
          </>
        ) : (
          <span className="text-rose-500">● backend unreachable</span>
        )}
      </div>
      <p className="text-xs text-slate-300 mt-2">
        MEDVISION is a college research prototype. It is not a certified medical device and must
        never be used for real clinical decisions.
      </p>
    </footer>
  );
}
