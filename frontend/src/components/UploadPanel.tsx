import { useRef, useState } from "react";

interface Props {
  onFileSelected: (file: File) => void;
  loading: boolean;
}

export function UploadPanel({ onFileSelected, loading }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = (files: FileList | null) => {
    if (files && files.length > 0) onFileSelected(files[0]);
  };

  return (
    <div
      className={`rounded-xl border-2 border-dashed p-8 text-center transition-colors cursor-pointer ${
        dragOver ? "border-sky-500 bg-sky-50" : "border-slate-300 bg-white"
      }`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
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
        onChange={(e) => handleFiles(e.target.files)}
      />
      {loading ? (
        <p className="text-slate-600 animate-pulse">Analyzing chest X-ray…</p>
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
