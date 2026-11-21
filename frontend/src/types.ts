// Mirrors the Pydantic models in medvision/pipeline.py and backend/app/schemas.py.
// Keep these in sync by hand for now; a generated OpenAPI client is a nice upgrade later.

export type FindingStatus = "present" | "possible" | "absent";

export interface PathStepOut {
  finding_node_id: string;
  finding_name: string;
  finding_prob: number;
  relation: string;
  weight: number;
  contribution: number;
}

export interface DifferentialOut {
  node_id: string;
  name: string;
  score: number;
  paths: PathStepOut[];
}

export interface FindingOut {
  node_id: string;
  label: string;
  name: string;
  prob: number; // calibrated (identity/unchanged for labels with no fitted temperature)
  raw_prob: number; // the classifier's original, uncalibrated output
  status: FindingStatus;
  region: string | null;
  region_share: number | null;
  heatmap_path: string | null;
}

export interface EvidenceFinding {
  id: string; // "F1", "F2", ...
  node_id: string;
  name: string;
  prob: number;
  status: FindingStatus;
  region: string | null;
}

export interface EvidenceDifferential {
  id: string; // "D1", "D2", ...
  node_id: string;
  name: string;
  score: number;
  supporting_fact_ids: string[];
}

export interface EvidenceKgFact {
  id: string; // "K1", "K2", ...
  text: string;
}

export interface EvidenceCase {
  id: string; // "C1", "C2", ...
  case_id: string;
  similarity: number;
  labels: string[];
  text: string;
}

export interface EvidenceBlock {
  findings: EvidenceFinding[];
  differentials: EvidenceDifferential[];
  kg_facts: EvidenceKgFact[];
  cases?: EvidenceCase[]; // absent on analyses stored before retrieval existed
}

export interface NodeHintOut {
  node_id: string;
  name: string;
  score: number;
}

export interface SimilarCaseOut {
  case_id: string;
  similarity: number;
  labels: string[];
  finding_node_ids: string[];
  split: string;
}

export interface RetrievalOut {
  node_hints: NodeHintOut[];
  similar_cases: SimilarCaseOut[];
}

export interface ReportItem {
  text: string;
  evidence_ids: string[];
}

export interface DifferentialItem {
  diagnosis: string;
  rationale: string;
  evidence_ids: string[];
}

export interface LLMReport {
  findings: ReportItem[];
  impression: ReportItem[];
  differentials: DifferentialItem[];
  limitations: string[];
  recommendation: string;
}

export type ReportSource = "llm" | "template";

export interface GroundingResult {
  score: number;
  unsupported: string[];
  contradictions: string[];
}

export interface AnalysisResult {
  id: string;
  created_at: number;
  findings: FindingOut[];
  differentials: DifferentialOut[];
  report: LLMReport;
  report_source: ReportSource;
  grounding: GroundingResult;
  evidence: EvidenceBlock;
  report_text: string;
  quality_warnings: string[]; // soft QC flags, e.g. an unusual aspect ratio; [] if none
  retrieval?: RetrievalOut | null; // null/absent when the alignment index isn't built
  timings_ms: Record<string, number>;
  disclaimer: string;
}

export interface AnalysisSummary {
  id: string;
  created_at: number;
  top_finding: string | null;
  top_differential: string | null;
}

export interface HealthOut {
  status: "ok";
  device: string;
  kg_nodes: number;
  kg_edges: number;
  retrieval_ready?: boolean;
  disclaimer: string;
}

export type FeedbackVerdict = "agree" | "disagree";

export interface FeedbackOut {
  id: number;
  analysis_id: string;
  node_id: string;
  verdict: FeedbackVerdict;
  comment: string | null;
  created_at: number;
}
