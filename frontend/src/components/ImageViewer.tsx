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
    <div className="bg-white rounded-xl border border-slate-200 p-3">
      <div className="relative w-full aspect-square bg-black rounded-lg overflow-hidden">
        <img
          src={originalImageUrl(analysisId)}
          alt="Chest X-ray"
          className="absolute inset-0 w-full h-full object-contain"
        />
        {selectedNodeId && (
          <img
            src={heatmapUrl(analysisId, selectedNodeId)}
            alt={`Grad-CAM heatmap for ${selectedLabel ?? selectedNodeId}`}
            className="absolute inset-0 w-full h-full object-contain"
            style={{ opacity }}
          />
        )}
      </div>
      <div className="flex items-center gap-3 mt-3 text-sm text-slate-600">
        <span className="whitespace-nowrap">
          {selectedNodeId ? `Heatmap: ${selectedLabel}` : "No finding selected"}
        </span>
        {selectedNodeId && (
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
            className="flex-1"
          />
        )}
      </div>
    </div>
  );
}
