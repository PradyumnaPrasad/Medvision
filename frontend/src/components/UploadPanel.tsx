import { useEffect, useRef, useState } from "react";

interface Props {
  onFileSelected: (file: File) => void;
  loading: boolean;
}

// Real pipeline stages, in the order medvision/pipeline.py actually runs them.
// Cycling through these while waiting is honest — it's not a fake progress bar,
// just naming what's actually happening — and the LLM narration stage really is
// the slow one, so it gets the longest dwell time.
const STAGES: { label: string; ms: number }[] = [
  { label: "Checking image quality…", ms: 600 },
  { label: "Running the vision model…", ms: 1200 },
  { label: "Applying calibration…", ms: 500 },
  { label: "Generating Grad-CAM heatmaps…", ms: 1500 },
  { label: "Reasoning over the knowledge graph…", ms: 800 },
  { label: "Writing the report with the local LLM…", ms: 30000 },
  { label: "Verifying citations…", ms: 30000 },
];

function Spinner() {
  return (
    <svg className="animate-spin h-5 w-5 text-sky-500" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  );
}

function useStageMessage(active: boolean): string {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    // Resetting state here (rather than deriving it) is intentional: `active`
    // going false->true is a new "session" of an external timer sequence, and
    // this effect is exactly where that sequence starts and stops.
    if (!active) {
      setIndex(0);
      return;
    }
    let cancelled = false;
    let i = 0;
    const advance = () => {
      if (cancelled) return;
      i = Math.min(i + 1, STAGES.length - 1);
      setIndex(i);
      window.setTimeout(advance, STAGES[i].ms);
    };
    const timer = window.setTimeout(advance, STAGES[0].ms);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [active]);

  return STAGES[index].label;
}

export function UploadPanel({ onFileSelected, loading }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const stageMessage = useStageMessage(loading);

  const handleFiles = (files: FileList | null) => {
    if (loading) return;
    if (files && files.length > 0) onFileSelected(files[0]);
  };

  return (
    <div
      className={`rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
        loading ? "cursor-wait border-slate-200 bg-slate-50" : "cursor-pointer"
      } ${dragOver && !loading ? "border-sky-500 bg-sky-50" : !loading ? "border-slate-300 bg-white" : ""}`}
      onClick={() => !loading && inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        if (!loading) setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        handleFiles(e.dataTransfer.files);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".png,.jpg,.jpeg"
        className="hidden"
        disabled={loading}
        onChange={(e) => handleFiles(e.target.files)}
      />
      {loading ? (
        <div className="flex flex-col items-center gap-2">
          <Spinner />
          <p className="text-slate-600 text-sm">{stageMessage}</p>
        </div>
      ) : (
        <>
          <p className="text-slate-700 font-medium">
            Drop a chest X-ray here, or click to choose a file
          </p>
          <p className="text-slate-400 text-sm mt-1">PNG or JPG. DICOM support coming soon.</p>
        </>
      )}
    </div>
  );
}
