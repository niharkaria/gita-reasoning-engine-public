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
