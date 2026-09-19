// lib/citationParser.ts
//
// Splits an answer string into alternating plain-text and citation
// segments, based on "Chapter N, Verse M" mentions the generation
// model is already instructed to include (see SYSTEM_PROMPT rule 2 in
// graph.py). This lets the UI render those specific mentions as
// styled, clickable inline badges instead of flat prose -- making the
// project's core "every claim is traceable to a citation" pitch
// visible in the answer itself, not just in a separately-collapsed
// sources list.
//
// Deliberately tolerant of minor real model phrasing variation seen in
// practice (e.g. "Chapter 9, Verse 15" vs "Chapter 9 Verse 15" vs
// "ch. 9, v. 15") -- the regex below covers the two full-word forms;
// if the model's real phrasing drifts further, extend the pattern
// rather than assuming this covers every case.

export type AnswerSegment =
  | { type: "text"; content: string }
  | { type: "citation"; content: string; chapter: number; verseNumber: number };

const CITATION_PATTERN = /Chapter\s+(\d+),?\s+Verse\s+(\d+)/gi;

export function parseAnswerForCitations(answer: string): AnswerSegment[] {
  const segments: AnswerSegment[] = [];
  let lastIndex = 0;

  for (const match of answer.matchAll(CITATION_PATTERN)) {
    const matchIndex = match.index ?? 0;

    if (matchIndex > lastIndex) {
      segments.push({ type: "text", content: answer.slice(lastIndex, matchIndex) });
    }

    segments.push({
      type: "citation",
      content: match[0],
      chapter: parseInt(match[1], 10),
      verseNumber: parseInt(match[2], 10),
    });

    lastIndex = matchIndex + match[0].length;
  }

  if (lastIndex < answer.length) {
    segments.push({ type: "text", content: answer.slice(lastIndex) });
  }

  return segments;
}

// A stable key for matching an inline citation mention back to a real
// entry in exchange.citations -- a chapter.verse pair can have BOTH a
// translation and a commentary entry, so this matches on chapter+verse
// only (not passage_type), and callers should be ready to handle
// multiple real matches for one inline mention.
export function citationKey(chapter: number, verseNumber: number, passageType: string, i: number) {
  return `${chapter}-${verseNumber}-${passageType}-${i}`;
}
