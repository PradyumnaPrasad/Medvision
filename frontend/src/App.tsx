import { useEffect, useState } from "react";
import { analyzeImage, getAnalysis, getHealth, listAnalyses } from "./api";
import { DifferentialsPanel } from "./components/DifferentialsPanel";
import { DisclaimerBanner } from "./components/DisclaimerBanner";
import { FindingsTable } from "./components/FindingsTable";
import { HistorySidebar } from "./components/HistorySidebar";
import { ImageViewer } from "./components/ImageViewer";
import { ReportPanel } from "./components/ReportPanel";
import { UploadPanel } from "./components/UploadPanel";
import type { AnalysisResult, AnalysisSummary } from "./types";

function App() {
  const [disclaimer, setDisclaimer] = useState(
    "Research prototype. Not for clinical use.",
  );
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [history, setHistory] = useState<AnalysisSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshHistory = () => {
    listAnalyses().then(setHistory).catch(() => {});
  };

  useEffect(() => {
    getHealth()
      .then((h) => setDisclaimer(h.disclaimer))
      .catch(() => setError("Could not reach the MEDVISION API. Is the backend running?"));
    refreshHistory();
  }, []);

  const handleFile = async (file: File) => {
    setLoading(true);
    setError(null);
    try {
      const analysis = await analyzeImage(file);
      applyResult(analysis);
      refreshHistory();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  const applyResult = (analysis: AnalysisResult) => {
    setResult(analysis);
    const topShown = analysis.findings.find((f) => f.status !== "absent");
    setSelectedNodeId(topShown?.node_id ?? null);
  };

  const handleHistorySelect = async (id: string) => {
    setLoading(true);
    setError(null);
    try {
      const analysis = await getAnalysis(id);
      applyResult(analysis);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load that analysis");
    } finally {
      setLoading(false);
    }
  };

  const selectedFinding = result?.findings.find((f) => f.node_id === selectedNodeId) ?? null;

  return (
    <div className="min-h-screen">
      <DisclaimerBanner text={disclaimer} />

      <header className="max-w-7xl mx-auto px-4 pt-6 pb-2">
        <h1 className="text-2xl font-bold text-slate-900">MEDVISION</h1>
        <p className="text-slate-500 text-sm">
          Knowledge-graph anchored vision-reasoning framework for explainable clinical diagnosis
        </p>
      </header>

      <main className="max-w-7xl mx-auto px-4 pb-10 grid grid-cols-1 lg:grid-cols-[260px_1fr_1fr] gap-4">
        <aside className="space-y-4">
          <UploadPanel onFileSelected={handleFile} loading={loading} />
          <HistorySidebar analyses={history} activeId={result?.id ?? null} onSelect={handleHistorySelect} />
        </aside>

        {error && (
          <div className="lg:col-span-2 bg-rose-50 border border-rose-200 text-rose-700 rounded-xl px-4 py-3 text-sm">
            {error}
          </div>
        )}

        {!error && !result && !loading && (
          <div className="lg:col-span-2 flex items-center justify-center text-slate-400 text-sm border border-dashed border-slate-200 rounded-xl">
            Upload a chest X-ray to see explainable findings and a differential diagnosis.
          </div>
        )}

        {result && (
          <>
            <div className="space-y-4">
              <ImageViewer
                analysisId={result.id}
                selectedNodeId={selectedNodeId}
                selectedLabel={selectedFinding?.name ?? null}
              />
              <FindingsTable
                analysisId={result.id}
                findings={result.findings}
                selectedNodeId={selectedNodeId}
                onSelect={setSelectedNodeId}
              />
            </div>
            <div className="space-y-4">
              <DifferentialsPanel differentials={result.differentials} />
              <ReportPanel reportText={result.report_text} timingsMs={result.timings_ms} />
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
