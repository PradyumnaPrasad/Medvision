import { useEffect, useRef, useState } from "react";
import {
  analyzeImage,
  getAnalysis,
  getFeedback,
  getHealth,
  listAnalyses,
  submitFeedback,
} from "./api";
import { DifferentialsPanel } from "./components/DifferentialsPanel";
import { DisclaimerBanner } from "./components/DisclaimerBanner";
import { FindingsTable } from "./components/FindingsTable";
import { HistorySidebar } from "./components/HistorySidebar";
import { AlertIcon, LogoMarkIcon } from "./components/icons";
import { ImageViewer } from "./components/ImageViewer";
import { QualityWarningBanner } from "./components/QualityWarningBanner";
import { ReportPanel } from "./components/ReportPanel";
import { SummaryCard } from "./components/SummaryCard";
import { SystemFooter } from "./components/SystemFooter";
import { ThemeToggle } from "./components/ThemeToggle";
import { UploadPanel } from "./components/UploadPanel";
import type { AnalysisResult, AnalysisSummary, FeedbackVerdict, HealthOut } from "./types";
import { useTheme } from "./useTheme";

function App() {
  const [theme, toggleTheme] = useTheme();
  const [health, setHealth] = useState<HealthOut | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [history, setHistory] = useState<AnalysisSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<Record<string, FeedbackVerdict>>({});
  const [feedbackError, setFeedbackError] = useState<string | null>(null);
  const currentId = useRef<string | null>(null);

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
    currentId.current = analysis.id;
    setResult(analysis);
    setFeedback({});
    setFeedbackError(null);
    const topShown = analysis.findings.find((f) => f.status !== "absent");
    setSelectedNodeId(topShown?.node_id ?? null);
    getFeedback(analysis.id)
      .then((items) => {
        if (currentId.current !== analysis.id) return; // user already moved on
        setFeedback(Object.fromEntries(items.map((f) => [f.node_id, f.verdict])));
      })
      .catch(() => {});
  };

  const handleFeedback = async (nodeId: string, verdict: FeedbackVerdict) => {
    if (!result) return;
    const analysisId = result.id;
    const previous = feedback[nodeId];
    if (previous === verdict) return;
    setFeedbackError(null);
    setFeedback((prev) => ({ ...prev, [nodeId]: verdict }));
    try {
      await submitFeedback(analysisId, nodeId, verdict);
    } catch (e) {
      if (currentId.current !== analysisId) return;
      setFeedback((prev) => {
        const next = { ...prev };
        if (previous) next[nodeId] = previous;
        else delete next[nodeId];
        return next;
      });
      setFeedbackError(e instanceof Error ? `Feedback not saved: ${e.message}` : "Feedback not saved");
    }
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

      <header className="max-w-7xl mx-auto px-4 pt-6 pb-3 w-full flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-slate-900 dark:bg-slate-950 ring-1 ring-white/10 flex items-center justify-center shrink-0">
          <LogoMarkIcon className="w-6 h-6" />
        </div>
        <div className="flex-1 min-w-0">
          <h1 className="text-xl font-bold text-slate-900 dark:text-white tracking-tight">
            MEDVISION
          </h1>
          <p className="text-slate-500 dark:text-slate-400 text-xs sm:text-sm truncate">
            Knowledge-graph anchored vision-reasoning framework for explainable clinical diagnosis
          </p>
        </div>
        <ThemeToggle theme={theme} onToggle={toggleTheme} />
      </header>

      <main className="max-w-7xl mx-auto px-4 pb-10 w-full flex-1 space-y-4">
        {!error && result && (
          <>
            <SummaryCard result={result} />
            {feedbackError && (
              <div
                role="alert"
                className="mv-card bg-rose-50/60 dark:bg-rose-500/10 border-rose-200 dark:border-rose-500/20 text-rose-700 dark:text-rose-300 px-4 py-2.5 text-sm flex items-center gap-2"
              >
                <AlertIcon className="w-4 h-4 shrink-0" />
                <span>{feedbackError}</span>
              </div>
            )}
          </>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-[260px_1fr_1fr] gap-4 items-start">
          <aside className="space-y-4">
            <UploadPanel onFileSelected={handleFile} loading={loading} />
            <HistorySidebar
              analyses={history}
              activeId={result?.id ?? null}
              onSelect={handleHistorySelect}
            />
          </aside>

          {error && (
            <div className="lg:col-span-2 mv-card mv-fade-in bg-rose-50/60 dark:bg-rose-500/10 border-rose-200 dark:border-rose-500/20 text-rose-700 dark:text-rose-300 px-4 py-3.5 text-sm flex items-start gap-2.5">
              <AlertIcon className="w-4 h-4 mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {!error && !result && !loading && (
            <div className="lg:col-span-2 flex flex-col items-center justify-center gap-3 text-center text-slate-400 text-sm border-2 border-dashed border-slate-200 dark:border-white/10 rounded-2xl min-h-[280px] px-8">
              <LogoMarkIcon className="w-10 h-10 opacity-40" />
              <p className="max-w-xs">
                Upload a chest X-ray to see explainable findings, a differential diagnosis, and a
                cited draft report.
              </p>
            </div>
          )}

          {!error && result && (
            <>
              <div className="space-y-4 mv-fade-in">
                {result.quality_warnings.length > 0 && (
                  <QualityWarningBanner warnings={result.quality_warnings} />
                )}
                <ImageViewer
                  analysisId={result.id}
                  selectedNodeId={selectedNodeId}
                  selectedLabel={selectedFinding?.name ?? null}
                />
                <FindingsTable
                  findings={result.findings}
                  feedback={feedback}
                  onFeedback={handleFeedback}
                  selectedNodeId={selectedNodeId}
                  onSelect={setSelectedNodeId}
                />
              </div>
              <div className="space-y-4 mv-fade-in">
                <DifferentialsPanel
                  differentials={result.differentials}
                  feedback={feedback}
                  onFeedback={handleFeedback}
                />
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
        </div>
      </main>

      <SystemFooter health={health} />
    </div>
  );
}

export default App;
