import type { AskResponse } from "@/lib/api";
import { CitationCard } from "@/components/CitationCard";

export function AnswerMessage({ exchange }: { exchange: AskResponse }) {
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
          <div className="space-y-3">
            <p className="text-xs uppercase tracking-wider text-brass font-body font-medium">
              Grounded in
            </p>
            {exchange.citations.map((c, i) => (
              <CitationCard key={`${c.chapter}-${c.verse_number}-${c.passage_type}-${i}`} citation={c} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
