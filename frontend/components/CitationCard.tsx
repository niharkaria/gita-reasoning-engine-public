import type { CitedPassage } from "@/lib/api";

export function CitationCard({ citation }: { citation: CitedPassage }) {
  const kindLabel = citation.passage_type === "translation" ? "Shlokārtha" : "Vivechan";

  return (
    <div className="relative border-l-2 border-brass bg-white/40 pl-5 pr-4 py-4 rounded-r-md">
      <div className="flex items-baseline gap-3 mb-2">
        <span
          className="inline-flex items-center justify-center min-w-[2.75rem] h-7 px-2 rounded-full border border-brass text-brass text-xs font-medium font-body tracking-wide"
          aria-label={`Chapter ${citation.chapter}, Verse ${citation.verse_number}`}
        >
          {citation.chapter}.{citation.verse_number}
        </span>
        <span className="text-xs uppercase tracking-wider text-ink-soft font-body">
          {kindLabel}
        </span>
      </div>
      <p className="font-gujarati text-[15px] leading-relaxed text-ink">{citation.text}</p>
    </div>
  );
}
