"use client";

import { useRef, useState } from "react";
import type { AskResponse } from "@/lib/api";
import { CitationCard } from "@/components/CitationCard";
import { parseAnswerForCitations, citationKey } from "@/lib/citationParser";

export function AnswerMessage({ exchange }: { exchange: AskResponse }) {
  const [showSources, setShowSources] = useState(false);
  const [highlightedKey, setHighlightedKey] = useState<string | null>(null);
  const cardRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  const segments = parseAnswerForCitations(exchange.answer);

  function handleInlineCitationClick(chapter: number, verseNumber: number) {
    // An inline "Chapter X, Verse Y" mention doesn't say whether it's
    // referring to the translation or commentary entry for that verse
    // -- a verse can have both. Jump to whichever real entry matches
    // first; if there's more than one, the first is treated as the
    // primary reference for that mention.
    const matchIndex = exchange.citations.findIndex(
      (c) => c.chapter === chapter && c.verse_number === verseNumber,
    );
    if (matchIndex === -1) return;

    const match = exchange.citations[matchIndex];
    const key = citationKey(match.chapter, match.verse_number, match.passage_type, matchIndex);

    setShowSources(true);
    setHighlightedKey(key);

    // Wait a tick for the sources section to actually render (it's
    // conditionally mounted on showSources) before trying to scroll to
    // a card that doesn't exist in the DOM yet.
    requestAnimationFrame(() => {
      cardRefs.current.get(key)?.scrollIntoView({ behavior: "smooth", block: "center" });
    });

    // Clear the highlight after a few seconds rather than leaving it
    // permanently -- it's meant to draw the eye to the just-clicked
    // reference, not to persistently mark it as "the current one."
    setTimeout(() => setHighlightedKey(null), 2500);
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-end">
        <div className="max-w-[85%] bg-ink text-stone rounded-2xl rounded-tr-sm px-5 py-3 font-body text-[15px]">
          {exchange.question}
        </div>
      </div>

      <div className="max-w-[85%]">
        <p className="font-display text-[17px] leading-relaxed text-ink mb-4">
          {segments.map((segment, i) =>
            segment.type === "text" ? (
              <span key={i}>{segment.content}</span>
            ) : (
              <button
                key={i}
                type="button"
                onClick={() => handleInlineCitationClick(segment.chapter, segment.verseNumber)}
                className="inline-flex items-center mx-0.5 px-2 py-0.5 rounded-full border border-brass bg-brass/10 text-brass text-[13px] font-body font-medium align-baseline hover:bg-brass/20 transition-colors"
              >
                {segment.chapter}.{segment.verseNumber}
              </button>
            ),
          )}
        </p>

        {exchange.citations.length > 0 && (
          <div>
            <button
              type="button"
              onClick={() => setShowSources((prev) => !prev)}
              className="text-xs uppercase tracking-wider text-brass font-body font-medium underline underline-offset-2"
              aria-expanded={showSources}
            >
              {showSources
                ? "Hide sources"
                : `Show sources (${exchange.citations.length})`}
            </button>

            {showSources && (
              <div className="space-y-3 mt-3">
                {exchange.citations.map((c, i) => {
                  const key = citationKey(c.chapter, c.verse_number, c.passage_type, i);
                  return (
                    <div
                      key={key}
                      ref={(el) => {
                        if (el) cardRefs.current.set(key, el);
                        else cardRefs.current.delete(key);
                      }}
                    >
                      <CitationCard citation={c} isHighlighted={highlightedKey === key} />
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}