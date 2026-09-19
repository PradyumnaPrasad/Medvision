import { useState } from "react";
import { heatmapUrl, originalImageUrl } from "../api";

interface Props {
  analysisId: string;
  selectedNodeId: string | null;
  selectedLabel: string | null;
}

export function ImageViewer({ analysisId, selectedNodeId, selectedLabel }: Props) {
  const [opacity, setOpacity] = useState(0.6);

  return (
    <div className="mv-card p-3">
      <div className="relative w-full aspect-square bg-slate-950 rounded-xl overflow-hidden ring-1 ring-inset ring-white/5">
        <img
          src={originalImageUrl(analysisId)}
          alt="Chest X-ray"
          className="absolute inset-0 w-full h-full object-contain"
        />
        {selectedNodeId && (
          <img
            key={selectedNodeId}
            src={heatmapUrl(analysisId, selectedNodeId)}
            alt={`Grad-CAM heatmap for ${selectedLabel ?? selectedNodeId}`}
            className="absolute inset-0 w-full h-full object-contain mv-fade-in"
            style={{ opacity }}
          />
        )}
      </div>
      <div className="flex items-center gap-3 mt-3 text-sm text-slate-600">
        <span className="whitespace-nowrap flex items-center gap-1.5">
          {selectedNodeId && (
            <span className="w-2 h-2 rounded-full bg-gradient-to-br from-amber-400 to-rose-500 shrink-0" />
          )}
          {selectedNodeId ? (
            <>
              Heatmap: <span className="font-medium text-slate-800">{selectedLabel}</span>
            </>
          ) : (
            "No finding selected"
          )}
        </span>
        {selectedNodeId && (
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
            className="flex-1 accent-sky-500"
          />
        )}
      </div>
    </div>
  );
}
