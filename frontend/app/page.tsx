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

// Browser-only conversation persistence: survives a page refresh, cleared
// explicitly by the Reset button. Not synced anywhere else (no backend,
// no cross-device history).
const HISTORY_STORAGE_KEY = "gita-reasoning-engine:exchanges";

// Browser-only theme preference persistence: survives a page refresh.
// Not cleared by Reset (Reset only clears the conversation, not display
// preferences).
const THEME_STORAGE_KEY = "gita-reasoning-engine:theme";

export default function Home() {
  const [exchanges, setExchanges] = useState<AskResponse[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [theme, setTheme] = useState<"light" | "dark">("dark");
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [themeLoaded, setThemeLoaded] = useState(false);

  // Bottom-of-page sentinel (used to scroll the loading indicator into view)
  const scrollRef = useRef<HTMLDivElement>(null);
  // Wrapper around the newest answer (used to scroll to the START of it)
  const lastAnswerRef = useRef<HTMLDivElement>(null);

  const prevExchangeCountRef = useRef(exchanges.length);
  const prevIsLoadingRef = useRef(isLoading);

  const hasConversation = exchanges.length > 0;
  // "Started" = the person has asked something. From that moment the
  // one-screen hero lock is released so the page can scroll normally.
  const hasStarted = hasConversation || isLoading || error !== null;

  // Restore a saved conversation on first load (survives refresh).
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(HISTORY_STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved) as AskResponse[];
        if (Array.isArray(parsed) && parsed.length > 0) {
          setExchanges(parsed);
          // Don't animate a scroll on initial restore -- jump straight there.
          prevExchangeCountRef.current = parsed.length;
        }
      }
    } catch {
      // Corrupt or unavailable storage -- just start fresh.
    } finally {
      setHistoryLoaded(true);
    }
  }, []);

  // Persist on every change, once the initial restore has happened (so we
  // don't immediately overwrite storage with an empty array before the
  // restore effect above has had a chance to run).
  useEffect(() => {
    if (!historyLoaded) return;
    try {
      if (exchanges.length > 0) {
        window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(exchanges));
      } else {
        window.localStorage.removeItem(HISTORY_STORAGE_KEY);
      }
    } catch {
      // Storage full/unavailable -- conversation still works, just won't persist.
    }
  }, [exchanges, historyLoaded]);

  useEffect(() => {
    const gotNewExchange = exchanges.length > prevExchangeCountRef.current;
    const startedLoading = !prevIsLoadingRef.current && isLoading;

    if (gotNewExchange) {
      // New answer: show the top of it, not the bottom.
      lastAnswerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    } else if (startedLoading) {
      // Question sent: make sure the loading indicator is visible.
      scrollRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    }

    prevExchangeCountRef.current = exchanges.length;
    prevIsLoadingRef.current = isLoading;
  }, [exchanges, isLoading]);

  useEffect(() => {
    if ("scrollRestoration" in window.history) {
      window.history.scrollRestoration = "manual";
    }
  }, []);

  // Restore saved theme preference on load. Note: initial state above is
  // always "dark", so a saved "light" preference causes a brief flash of
  // dark theme on first paint before this effect runs -- a real limitation
  // of client-only persistence without a server-side cookie, not a bug.
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(THEME_STORAGE_KEY);
      if (saved === "light" || saved === "dark") {
        setTheme(saved);
      }
    } catch {
      // Storage unavailable -- keep the "dark" default.
    } finally {
      setThemeLoaded(true);
    }
  }, []);

  // Applies the chosen theme by setting data-theme on <html>, which the
  // dark-theme override rules in globals.css key off of.
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  // Persist theme choice whenever it changes, once the initial restore has
  // happened. The `themeLoaded` gate matters here for the same reason it
  // matters for history above: without it, this effect fires once on mount
  // with the default "dark" BEFORE the restore effect's setTheme() above has
  // taken effect, silently overwriting a saved "light" preference back to
  // "dark" in storage on every single page load. Confirmed as a real bug
  // (not hypothetical) via manual testing 2026-09-20 before this gate was
  // added. Deliberately NOT cleared by resetToHome() -- Reset clears the
  // conversation, not display preferences.
  useEffect(() => {
    if (!themeLoaded) return;
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // Storage unavailable -- theme still works, just won't persist.
    }
  }, [theme, themeLoaded]);

  function toggleTheme() {
    setTheme((prev) => (prev === "light" ? "dark" : "light"));
  }

  async function submitQuestion(question: string) {
    if (!question.trim() || isLoading) return;
    setIsLoading(true);
    setError(null);
    setInput("");
    setPendingQuestion(question.trim());

    try {
      const result = await askQuestion(question);
      setExchanges((prev) => [...prev, result]);
      setPendingQuestion(null);
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
    if (isLoading) return;
    setExchanges([]);
    setError(null);
    setPendingQuestion(null);
    try {
      window.localStorage.removeItem(HISTORY_STORAGE_KEY);
    } catch {
      // Ignore -- nothing to clean up if storage isn't available.
    }
    window.scrollTo({ top: 0 });
  }

  return (
    // "is-hero" locks the layout to exactly one viewport (see the
    // ONE-SCREEN HERO block in globals.css). It is removed as soon as a
    // question is submitted, so the loading state and the answer can scroll.
    <div className={`app-wrapper${hasStarted ? "" : " is-hero"}`}>
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
            {hasStarted && (
              <button onClick={resetToHome} className="reset-btn" disabled={isLoading}>
                Reset
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="hero-content">
        {/* Mandala backdrop: always mounted now, not just on the landing
            screen. `is-subtle` (added once a conversation has started)
            fades it down to a faint watermark via the CSS rules in
            globals.css, and the backdrop itself is `position: fixed` so
            it stays centered on screen while a long conversation scrolls,
            rather than only appearing once partway down the page. */}
        <div
          className={`hero-chakra-backdrop${hasStarted ? " is-subtle" : ""}`}
          aria-hidden="true"
        >
          <Image
            src="/chakra-bg-swirl-web.webp"
            alt=""
            fill
            priority
            sizes="(max-width: 714px) 140vw, 1000px"
            className="hero-chakra-swirl"
          />
          <Image
            src="/chakra-core-static-web.webp"
            alt=""
            fill
            priority
            sizes="(max-width: 714px) 140vw, 1000px"
            className="hero-chakra-core"
          />
        </div>

        {!hasStarted && <div className="hero-text-scrim" aria-hidden="true" />}

        <div className="hero-inner">
          {!hasStarted ? (
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

              <h1 className="font-cinzel hero-h1">Gita Reasoning Engine</h1>

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
            <div className="chat-view">
              {exchanges.map((exchange, i) => (
                <div
                  key={i}
                  className="answer-item"
                  ref={i === exchanges.length - 1 ? lastAnswerRef : undefined}
                >
                  <AnswerMessage exchange={exchange} />
                </div>
              ))}

              {pendingQuestion && (
                <div className="pending-question">{pendingQuestion}</div>
              )}

              {isLoading && (
                <div className="loading-pill" role="status">
                  <span className="loading-dot" />
                  Consulting the Vallabhacharya Subodhini &amp; Pushtimarg commentaries...
                </div>
              )}

              {error && (
                <div role="alert" className="error-box">
                  {error}
                </div>
              )}
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