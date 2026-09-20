"use client";

import { useState } from "react";
import type { CitedPassage } from "@/lib/api";

export function CitationCard({
  citation,
  isHighlighted = false,
}: {
  citation: CitedPassage;
  isHighlighted?: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const kindLabel = citation.passage_type === "translation" ? "Shlokārtha" : "Vivechan";

  return (
    <div className={`cite-card${isHighlighted ? " is-highlighted" : ""}`}>
      <div className="cite-card-ornament" aria-hidden="true">✦</div>

      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        className="cite-card-head"
        aria-expanded={expanded}
      >
        <span
          className="cite-chip cite-chip-static"
          aria-label={`Chapter ${citation.chapter}, Verse ${citation.verse_number}`}
        >
          {citation.chapter}.{citation.verse_number}
        </span>
        <span className="cite-card-hint">
          {expanded ? "Hide commentary" : "Tap to view commentary"}
        </span>
        <span className={`cite-card-chevron${expanded ? " is-open" : ""}`} aria-hidden="true">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </span>
      </button>

      <p className="cite-card-sanskrit">{citation.sanskrit_text}</p>

      {expanded && (
        <div className="cite-card-more">
          <span className="cite-card-kind">{kindLabel}</span>
          <p className="cite-card-text">{citation.text}</p>
        </div>
      )}
    </div>
  );
}