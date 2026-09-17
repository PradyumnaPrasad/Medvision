import type { EvidenceBlock } from "../types";

interface Props {
  text: string;
  evidenceIds: string[];
  evidence: EvidenceBlock;
}

function resolveEvidenceLabel(id: string, evidence: EvidenceBlock): string {
  const finding = evidence.findings.find((f) => f.id === id);
  if (finding) {
    return `${id}: ${finding.name} (${finding.status}, p=${(finding.prob * 100).toFixed(0)}%)`;
  }
  const differential = evidence.differentials.find((d) => d.id === id);
  if (differential) {
    return `${id}: ${differential.name} (score=${(differential.score * 100).toFixed(0)}%)`;
  }
  const fact = evidence.kg_facts.find((k) => k.id === id);
  if (fact) return `${id}: ${fact.text}`;
  return `${id}: unknown evidence`;
}

/** Renders a sentence followed by small superscript citation chips (F1, D2, K3, ...).
 * Hovering a chip shows the actual evidence it points to, via the native title tooltip —
 * this is what makes the LLM's report auditable instead of just prose. */
export function CitedText({ text, evidenceIds, evidence }: Props) {
  return (
    <span>
      {text}{" "}
      {evidenceIds.map((id) => (
        <sup key={id} title={resolveEvidenceLabel(id, evidence)}>
          <span className="ml-0.5 inline-block px-1 rounded bg-indigo-100 text-indigo-700 text-[10px] font-mono cursor-help align-super">
            {id}
          </span>
        </sup>
      ))}
    </span>
  );
}
