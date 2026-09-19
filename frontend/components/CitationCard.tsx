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
    <div
      className={`relative border-l-2 border-brass bg-white/40 pl-5 pr-4 py-4 rounded-r-md transition-shadow duration-500 ${
        isHighlighted ? "ring-2 ring-brass shadow-md" : ""
      }`}
    >
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        className="flex items-baseline gap-3 mb-2 w-full text-left"
        aria-expanded={expanded}
      >
        <span
          className="inline-flex items-center justify-center min-w-[2.75rem] h-7 px-2 rounded-full border border-brass text-brass text-xs font-medium font-body tracking-wide"
          aria-label={`Chapter ${citation.chapter}, Verse ${citation.verse_number}`}
        >
          {citation.chapter}.{citation.verse_number}
        </span>
        <span className="text-xs uppercase tracking-wider text-ink-soft font-body">
          {expanded ? "Hide commentary" : "Tap to view commentary"}
        </span>
      </button>

      <p className="font-gujarati text-[15px] leading-relaxed text-ink">
        {citation.sanskrit_text}
      </p>

      {expanded && (
        <div className="mt-3 pt-3 border-t border-brass/30">
          <span className="text-xs uppercase tracking-wider text-ink-soft font-body block mb-1">
            {kindLabel}
          </span>
          <p className="font-gujarati text-[15px] leading-relaxed text-ink">{citation.text}</p>
        </div>
      )}
    </div>
  );
}