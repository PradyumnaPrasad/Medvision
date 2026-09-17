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
  prob: number;
  status: FindingStatus;
  region: string | null;
  region_share: number | null;
  heatmap_path: string | null;
}

export interface AnalysisResult {
  id: string;
  created_at: number;
  findings: FindingOut[];
  differentials: DifferentialOut[];
  report_text: string;
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
  disclaimer: string;
}
