"use client";

import { useState } from "react";
import type { AskResponse } from "@/lib/api";
import { CitationCard } from "@/components/CitationCard";

export function AnswerMessage({ exchange }: { exchange: AskResponse }) {
  const [showSources, setShowSources] = useState(false);

  return (
    <div className="space-y-4">
      <div className="flex justify-end">
        <div className="max-w-[85%] bg-ink text-stone rounded-2xl rounded-tr-sm px-5 py-3 font-body text-[15px]">
          {exchange.question}
        </div>
      </div>

      <div className="max-w-[85%]">
        <p className="font-display text-[17px] leading-relaxed text-ink mb-4">
          {exchange.answer}
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
                {exchange.citations.map((c, i) => (
                  <CitationCard
                    key={`${c.chapter}-${c.verse_number}-${c.passage_type}-${i}`}
                    citation={c}
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}