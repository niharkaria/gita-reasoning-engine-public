"use client";

import { useState, useRef, useEffect, type FormEvent } from "react";
import Image from "next/image";
import { askQuestion, ApiError, type AskResponse } from "@/lib/api";
import { AnswerMessage } from "@/components/AnswerMessage";

const EXAMPLE_QUESTIONS = [
  {
    icon: "🪷",
    title: "What happens to the soul when the body dies?",
    query: "What happens to the soul when the body dies according to Pushtimarg?",
  },
  {
    icon: "🔥",
    title: "Who is the true enjoyer of all sacrifices?",
    query: "Who is recognized as the supreme enjoyer of all sacrifices (Yajnas)?",
  },
  {
    icon: "✨",
    title: "Describe Krishna's universal form (Vishwaroopa).",
    query: "How does Vallabhacharya explain Krishna's Vishwaroopa (universal form)?",
  },
];

export default function Home() {
  const [exchanges, setExchanges] = useState<AskResponse[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const scrollRef = useRef<HTMLDivElement>(null);

  const prevExchangeCountRef = useRef(exchanges.length);
  const prevIsLoadingRef = useRef(isLoading);

  const hasConversation = exchanges.length > 0;

  useEffect(() => {
    const gotNewExchange = exchanges.length > prevExchangeCountRef.current;
    const justFinishedLoading = prevIsLoadingRef.current && !isLoading;

    if (gotNewExchange || justFinishedLoading) {
      scrollRef.current?.scrollIntoView({ behavior: "smooth" });
    }

    prevExchangeCountRef.current = exchanges.length;
    prevIsLoadingRef.current = isLoading;
  }, [exchanges, isLoading]);

  useEffect(() => {
    if ("scrollRestoration" in window.history) {
      window.history.scrollRestoration = "manual";
    }
  }, []);

  // Applies the chosen theme by setting data-theme on <html>, which the
  // dark-theme override rules in globals.css key off of.
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  function toggleTheme() {
    setTheme((prev) => (prev === "light" ? "dark" : "light"));
  }

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

  function resetToHome() {
    setExchanges([]);
    setError(null);
  }

  return (
    // "is-hero" locks the layout to exactly one viewport (see the
    // ONE-SCREEN HERO block at the end of globals.css). Once a
    // conversation starts, the class is removed and the page scrolls
    // normally so long answers stay readable.
    <div className={`app-wrapper${hasConversation ? "" : " is-hero"}`}>
      <header className="navbar-header">
        <div className="navbar-inner">
          <div className="nav-left">
            <div className="nav-logo-box" onClick={resetToHome} style={{ cursor: "pointer" }}>
              <Image
                src="/sampraday-logo.png"
                alt="Sampraday emblem"
                width={34}
                height={34}
                priority
              />
            </div>
            <div className="nav-title-row">
              <span className="font-cinzel nav-title">GITA REASONING</span>
              <span className="nav-tag">Pushtimarg Corpus</span>
              <span className="nav-info" tabIndex={0}>
                <span className="nav-info-icon" aria-hidden="true">i</span>
                <span className="nav-info-tooltip" role="tooltip">
                  Grounded strictly in Shuddhadvaita Brahmavada
                </span>
              </span>
            </div>
          </div>

          <div className="nav-actions">
            <button
              onClick={toggleTheme}
              className="reset-btn"
              aria-label="Toggle light/dark theme"
            >
              {theme === "light" ? "🌙 Dark" : "☀️ Light"}
            </button>
            {hasConversation && (
              <button onClick={resetToHome} className="reset-btn">
                Reset
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="hero-content">
        {!hasConversation && (
          <div className="hero-chakra-backdrop" aria-hidden="true">
            <Image
              src="/chakra-bg-swirl-web.webp"
              alt=""
              fill
              priority
              className="hero-chakra-swirl"
            />
            <Image
              src="/chakra-core-static-web.webp"
              alt=""
              fill
              priority
              className="hero-chakra-core"
            />
          </div>
        )}

        {!hasConversation && (
          <div className="hero-text-scrim" aria-hidden="true" />
        )}

        <div className="hero-inner">
          {!hasConversation ? (
            <>
              <div className="mantra-pill">
                <span style={{ color: "#c59b27", fontSize: 13 }}>✦</span>
                <span className="font-playfair mantra-text">॥ श्रीकृष्णः शरणं मम ॥</span>
                <span style={{ color: "#c59b27", fontSize: 13 }}>✦</span>
              </div>

              <div className="center-logo-wrap">
                <div className="center-logo-circle">
                  <Image
                    src="/sampraday-logo.png"
                    alt="Sampraday emblem"
                    width={90}
                    height={90}
                    priority
                  />
                </div>
              </div>

              <h1 className="font-cinzel hero-h1">
                Gita Reasoning Engine
              </h1>

              <p className="hero-desc">
                Direct retrieval from canonical Pushtimarg commentaries. Grounded strictly in Mahaprabhu Shri Vallabhacharya's Darshana.
              </p>

              <div className="vasudeva-row">
                <span className="vasudeva-line" />
                <span className="font-playfair vasudeva-text">॥ ॐ नमो भगवते वासुदेवाय ॥</span>
                <span className="vasudeva-line" />
              </div>

              <div className="prompts-container">
                {EXAMPLE_QUESTIONS.map((item) => (
                  <button
                    key={item.title}
                    onClick={() => void submitQuestion(item.query)}
                    className="prompt-btn"
                  >
                    <div className="prompt-left">
                      <div className="prompt-icon">{item.icon}</div>
                      <span className="prompt-text">{item.title}</span>
                    </div>
                    <span className="prompt-arrow-badge" aria-hidden="true">
                      <svg
                        width="13"
                        height="13"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <polyline points="9 18 15 12 9 6" />
                      </svg>
                    </span>
                  </button>
                ))}
              </div>

              <div className="shloka-box">
                <div className="shloka-ornament" aria-hidden="true">✦</div>
                <p className="font-playfair shloka-verse">
                  कर्मण्येवाधिकारस्ते मा फलेषु कदाचन।
                </p>
                <p className="shloka-trans">
                  “Your divine entitlement is solely to action performed in loving service, never to the fruits that follow.”
                </p>
                <div className="shloka-divider" aria-hidden="true" />
                <div className="shloka-ref">
                  Shrimad Bhagavad Gita • Adhyaya 2, Shloka 47
                </div>
              </div>
            </>
          ) : (
            <div style={{ width: "100%", display: "flex", flexDirection: "column", gap: 20 }}>
              {exchanges.map((exchange, i) => (
                <AnswerMessage key={i} exchange={exchange} />
              ))}
            </div>
          )}

          {isLoading && (
            <div
              style={{
                marginTop: 24,
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
                padding: "8px 20px",
                borderRadius: 9999,
                background: "#ffffff",
                border: "1.5px solid #d4af37",
                color: "#6b1717",
                fontSize: 12.5,
                fontWeight: 600,
                boxShadow: "0 2px 8px rgba(0,0,0,0.04)",
              }}
            >
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: "#c59b27",
                  display: "inline-block",
                }}
              />
              Consulting the Vallabhacharya Subodhini & Pushtimarg commentaries...
            </div>
          )}

          {error && (
            <div
              role="alert"
              style={{
                marginTop: 20,
                padding: "12px 18px",
                borderRadius: 12,
                border: "1px solid #f87171",
                background: "#fef2f2",
                color: "#991b1b",
                fontSize: 13,
                fontWeight: 600,
                textAlign: "center",
              }}
            >
              {error}
            </div>
          )}

          <div ref={scrollRef} />
        </div>
      </main>

      <footer className="footer-dock">
        <form onSubmit={handleSubmit} className="dock-form">
          <div className="dock-container">
            <span style={{ fontSize: 16 }}>🪷</span>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about a verse, Pushtimarg teaching, or leela theme..."
              disabled={isLoading}
              className="dock-input"
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="font-cinzel dock-submit"
            >
              ASK
            </button>
          </div>
        </form>
      </footer>
    </div>
  );
}