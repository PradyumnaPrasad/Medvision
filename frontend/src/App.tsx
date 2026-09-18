import { useEffect, useState } from "react";
import { analyzeImage, getAnalysis, getHealth, listAnalyses } from "./api";
import { DifferentialsPanel } from "./components/DifferentialsPanel";
import { DisclaimerBanner } from "./components/DisclaimerBanner";
import { FindingsTable } from "./components/FindingsTable";
import { HistorySidebar } from "./components/HistorySidebar";
import { ImageViewer } from "./components/ImageViewer";
import { QualityWarningBanner } from "./components/QualityWarningBanner";
import { ReportPanel } from "./components/ReportPanel";
import { SystemFooter } from "./components/SystemFooter";
import { UploadPanel } from "./components/UploadPanel";
import type { AnalysisResult, AnalysisSummary, HealthOut } from "./types";

function App() {
  const [health, setHealth] = useState<HealthOut | null>(null);
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
      .then(setHealth)
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
    <div className="min-h-screen flex flex-col">
      <DisclaimerBanner
        text={health?.disclaimer ?? "Research prototype. Not for clinical use."}
      />

      <header className="max-w-7xl mx-auto px-4 pt-6 pb-2 w-full">
        <h1 className="text-2xl font-bold text-slate-900">MEDVISION</h1>
        <p className="text-slate-500 text-sm">
          Knowledge-graph anchored vision-reasoning framework for explainable clinical diagnosis
        </p>
      </header>

      <main className="max-w-7xl mx-auto px-4 pb-10 w-full flex-1 grid grid-cols-1 lg:grid-cols-[260px_1fr_1fr] gap-4 items-start">
        <aside className="space-y-4">
          <UploadPanel onFileSelected={handleFile} loading={loading} />
          <HistorySidebar
            analyses={history}
            activeId={result?.id ?? null}
            onSelect={handleHistorySelect}
          />
        </aside>

        {error && (
          <div className="lg:col-span-2 bg-rose-50 border border-rose-200 text-rose-700 rounded-xl px-4 py-3 text-sm flex items-start gap-2">
            <span>⚠️</span>
            <span>{error}</span>
          </div>
        )}

        {!error && !result && !loading && (
          <div className="lg:col-span-2 flex items-center justify-center text-slate-400 text-sm border border-dashed border-slate-200 rounded-xl min-h-[200px]">
            Upload a chest X-ray to see explainable findings and a differential diagnosis.
          </div>
        )}

        {!error && result && (
          <>
            <div className="space-y-4">
              {result.quality_warnings.length > 0 && (
                <QualityWarningBanner warnings={result.quality_warnings} />
              )}
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
              <ReportPanel
                report={result.report}
                report_source={result.report_source}
                grounding={result.grounding}
                evidence={result.evidence}
                timings_ms={result.timings_ms}
              />
            </div>
          </>
        )}
      </main>

      <SystemFooter health={health} />
    </div>
  );
}

export default App;
