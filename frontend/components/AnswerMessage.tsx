"use client";

import { Fragment, useRef, useState, type ReactNode } from "react";
import type { AskResponse } from "@/lib/api";
import { CitationCard } from "@/components/CitationCard";
import { parseAnswerForCitations, citationKey } from "@/lib/citationParser";

type CitationClick = (chapter: number, verseNumber: number) => void;

// ---------------------------------------------------------------------
// Inline rendering: **bold**, *italic*, and "Chapter N, Verse M"
// mentions turned into clickable citation chips. Nothing here uses a
// markdown library -- the model only emits a small, predictable subset
// (paragraphs, bullets, bold), so a tiny parser is enough.
// ---------------------------------------------------------------------

// Plain text -> text + citation chips. Any stray "*" left over from
// unbalanced markdown is dropped so it never shows up on screen.
function renderWithCitations(text: string, onCite: CitationClick, keyPrefix: string): ReactNode[] {
  return parseAnswerForCitations(text).map((segment, i) =>
    segment.type === "text" ? (
      <Fragment key={`${keyPrefix}-t${i}`}>{segment.content.replace(/\*+/g, "")}</Fragment>
    ) : (
      <button
        key={`${keyPrefix}-c${i}`}
        type="button"
        className="cite-chip"
        onClick={() => onCite(segment.chapter, segment.verseNumber)}
      >
        {segment.chapter}.{segment.verseNumber}
      </button>
    ),
  );
}

function renderItalics(text: string, onCite: CitationClick, keyPrefix: string): ReactNode[] {
  return text.split(/(\*[^*\s][^*]*\*)/g).flatMap((part, i) => {
    if (part.length > 2 && part.startsWith("*") && part.endsWith("*")) {
      return [
        <em key={`${keyPrefix}-e${i}`}>
          {renderWithCitations(part.slice(1, -1), onCite, `${keyPrefix}-e${i}`)}
        </em>,
      ];
    }
    return renderWithCitations(part, onCite, `${keyPrefix}-x${i}`);
  });
}

function renderInline(text: string, onCite: CitationClick, keyPrefix: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).flatMap((part, i) => {
    if (part.length > 4 && part.startsWith("**") && part.endsWith("**")) {
      return [
        <strong key={`${keyPrefix}-b${i}`}>
          {renderItalics(part.slice(2, -2), onCite, `${keyPrefix}-b${i}`)}
        </strong>,
      ];
    }
    return renderItalics(part, onCite, `${keyPrefix}-n${i}`);
  });
}

// ---------------------------------------------------------------------
// Block parsing: paragraphs, bullet lists, headings.
// ---------------------------------------------------------------------

type Block =
  | { kind: "p"; text: string }
  | { kind: "h"; text: string }
  | { kind: "ul"; items: string[] };

function parseBlocks(answer: string): Block[] {
  const blocks: Block[] = [];
  let para: string[] = [];
  let list: string[] = [];

  const flushPara = () => {
    if (para.length > 0) {
      blocks.push({ kind: "p", text: para.join(" ") });
      para = [];
    }
  };
  const flushList = () => {
    if (list.length > 0) {
      blocks.push({ kind: "ul", items: list });
      list = [];
    }
  };

  for (const rawLine of answer.replace(/\r/g, "").split("\n")) {
    const line = rawLine.trim();

    if (!line) {
      flushPara();
      flushList();
      continue;
    }

    const heading = line.match(/^#{1,6}\s+(.*)$/);
    // "* text" / "- text" is a bullet; "**bold**" (no space after the
    // first asterisk) is NOT, so it falls through to a paragraph.
    const bullet = line.match(/^[*\-•]\s+(.*)$/);

    if (heading) {
      flushPara();
      flushList();
      blocks.push({ kind: "h", text: heading[1] });
    } else if (bullet) {
      flushPara();
      list.push(bullet[1]);
    } else {
      flushList();
      para.push(line);
    }
  }

  flushPara();
  flushList();
  return blocks;
}

export function AnswerMessage({ exchange }: { exchange: AskResponse }) {
  const [showSources, setShowSources] = useState(false);
  const [highlightedKey, setHighlightedKey] = useState<string | null>(null);
  const cardRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  const blocks = parseBlocks(exchange.answer);

  function handleInlineCitationClick(chapter: number, verseNumber: number) {
    // An inline "Chapter X, Verse Y" mention doesn't say whether it's
    // referring to the translation or commentary entry for that verse
    // -- a verse can have both. Jump to whichever real entry matches
    // first.
    const matchIndex = exchange.citations.findIndex(
      (c) => c.chapter === chapter && c.verse_number === verseNumber,
    );
    if (matchIndex === -1) return;

    const match = exchange.citations[matchIndex];
    const key = citationKey(match.chapter, match.verse_number, match.passage_type, matchIndex);

    setShowSources(true);
    setHighlightedKey(key);

    // Wait a tick for the sources section to render before scrolling
    // to a card that isn't in the DOM yet.
    requestAnimationFrame(() => {
      cardRefs.current.get(key)?.scrollIntoView({ behavior: "smooth", block: "center" });
    });

    setTimeout(() => setHighlightedKey(null), 2500);
  }

  return (
    <div className="answer-msg">
      {/* Same look as the pending-question bubble shown while loading,
          so there is no visual jump when the answer arrives. */}
      <div className="pending-question">{exchange.question}</div>

      <div className="answer-body">
        {blocks.map((block, i) => {
          if (block.kind === "h") {
            return (
              <h3 key={i} className="answer-heading">
                {renderInline(block.text, handleInlineCitationClick, `h${i}`)}
              </h3>
            );
          }
          if (block.kind === "ul") {
            return (
              <ul key={i} className="answer-list">
                {block.items.map((item, j) => (
                  <li key={j}>{renderInline(item, handleInlineCitationClick, `l${i}-${j}`)}</li>
                ))}
              </ul>
            );
          }
          return (
            <p key={i} className="answer-para">
              {renderInline(block.text, handleInlineCitationClick, `p${i}`)}
            </p>
          );
        })}

        {exchange.citations.length > 0 && (
          <div className="sources-wrap">
            <button
              type="button"
              onClick={() => setShowSources((prev) => !prev)}
              className="sources-toggle"
              aria-expanded={showSources}
            >
              {showSources ? "Hide sources" : `Show sources (${exchange.citations.length})`}
            </button>

            {showSources && (
              <div className="sources-list">
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
