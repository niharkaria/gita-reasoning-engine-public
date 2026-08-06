"use client";

import { useState, useRef, useEffect, type FormEvent } from "react";
import { askQuestion, ApiError, type AskResponse } from "@/lib/api";
import { AnswerMessage } from "@/components/AnswerMessage";

const EXAMPLE_QUESTIONS = [
  "What happens to the soul when the body dies?",
  "Who is the true enjoyer of all sacrifices?",
  "Describe Krishna's universal form.",
];

export default function Home() {
  const [exchanges, setExchanges] = useState<AskResponse[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [exchanges, isLoading]);

  async function submitQuestion(question: string) {
    if (!question.trim() || isLoading) return;
    setIsLoading(true);
    setError(null);
    setInput("");

    try {
      const result = await askQuestion(question);
      setExchanges((prev) => [...prev, result]);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Couldn't reach the reasoning engine. Is the API running?",
      );
    } finally {
      setIsLoading(false);
    }
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    void submitQuestion(input);
  }

  return (
    <>
      <header className="border-b border-brass-soft bg-stone/95 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-2xl mx-auto px-6 py-5">
          <h1 className="font-display text-2xl text-oxblood tracking-tight">
            Gita Reasoning Engine
          </h1>
          <p className="text-sm text-ink-soft font-body mt-1">
            Answers are grounded strictly in one accepted Pushtimarg commentary —
            never general knowledge.
          </p>
        </div>
      </header>

      <main className="flex-1 max-w-2xl w-full mx-auto px-6 py-8">
        {exchanges.length === 0 ? (
          <div className="pt-8">
            <p className="text-ink-soft font-body text-sm mb-4">Try asking:</p>
            <div className="space-y-2">
              {EXAMPLE_QUESTIONS.map((q) => (
                <button
                  key={q}
                  onClick={() => void submitQuestion(q)}
                  className="block w-full text-left px-4 py-3 rounded-lg border border-brass-soft bg-white/40 hover:bg-white/70 hover:border-brass transition-colors font-body text-[15px] text-ink"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-10">
            {exchanges.map((exchange, i) => (
              <AnswerMessage key={i} exchange={exchange} />
            ))}
          </div>
        )}

        {isLoading && (
          <div className="mt-8 flex items-center gap-2 text-ink-soft font-body text-sm">
            <span className="inline-block w-2 h-2 rounded-full bg-brass animate-pulse" />
            Consulting the commentary…
          </div>
        )}

        {error && (
          <div
            role="alert"
            className="mt-8 px-4 py-3 rounded-lg border border-oxblood-dim bg-oxblood/5 text-oxblood text-sm font-body"
          >
            {error}
          </div>
        )}

        <div ref={scrollRef} />
      </main>

      <footer className="border-t border-brass-soft bg-stone/95 backdrop-blur-sm sticky bottom-0">
        <form onSubmit={handleSubmit} className="max-w-2xl mx-auto px-6 py-4 flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about a teaching, verse, or theme…"
            disabled={isLoading}
            className="flex-1 px-4 py-3 rounded-lg border border-brass-soft bg-white/60 font-body text-[15px] text-ink placeholder:text-ink-soft/60 disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="px-5 py-3 rounded-lg bg-oxblood text-stone font-body text-sm font-medium hover:bg-oxblood-dim transition-colors disabled:opacity-40 disabled:hover:bg-oxblood"
          >
            Ask
          </button>
        </form>
      </footer>
    </>
  );
}
