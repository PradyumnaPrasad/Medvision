import type { AnalysisResult, AnalysisSummary, HealthOut } from "./types";

const BASE = "/api/v1";

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      // response body wasn't JSON; fall back to statusText
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthOut> {
  const response = await fetch(`${BASE}/health`);
  return handleResponse(response);
}

export async function analyzeImage(file: File): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${BASE}/analyze`, { method: "POST", body: formData });
  return handleResponse(response);
}

export async function listAnalyses(limit = 20): Promise<AnalysisSummary[]> {
  const response = await fetch(`${BASE}/analyses?limit=${limit}`);
  return handleResponse(response);
}

export async function getAnalysis(id: string): Promise<AnalysisResult> {
  const response = await fetch(`${BASE}/analyses/${id}`);
  return handleResponse(response);
}

export function heatmapUrl(analysisId: string, nodeId: string): string {
  return `${BASE}/analyses/${analysisId}/heatmap/${nodeId}.png`;
}

export function originalImageUrl(analysisId: string): string {
  return `${BASE}/analyses/${analysisId}/original.png`;
}

export async function submitFeedback(
  analysisId: string,
  nodeId: string,
  verdict: "agree" | "disagree",
  comment?: string,
): Promise<void> {
  const response = await fetch(`${BASE}/analyses/${analysisId}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ node_id: nodeId, verdict, comment }),
  });
  await handleResponse(response);
}
