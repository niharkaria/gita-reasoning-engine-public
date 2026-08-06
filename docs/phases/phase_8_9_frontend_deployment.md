# Phases 8-9: Frontend and Deployment

## What was built

**Phase 8 — Frontend**
- `src/gita_engine/api/`: the FastAPI layer that was missing before this
  phase. `main.py` (app factory + CORS), `routes.py` (`/health`, `/ask`),
  `schemas.py` (request/response Pydantic models). Wraps
  `answer_question()` from Phase 6 as a real HTTP endpoint.
- `frontend/`: a Next.js 16 (App Router, TypeScript, Tailwind) chat
  interface. Deliberately designed rather than templated — see the
  design rationale below.

**Phase 9 — Deployment**
- Decided against AWS/Azure (real cost/complexity for a portfolio
  project) in favor of Docker Compose — genuinely free, runs on any
  machine with Docker.
- `frontend/Dockerfile`: multi-stage build using Next.js standalone
  output mode.
- `docker/docker-compose.yml`: now runs Postgres + API + frontend
  together with `make up`.

## Design rationale (frontend)

Consulted the frontend-design skill before building rather than
defaulting to a template. The subject — Sanskrit shlokas + Gujarati
commentary + citations — gave real material to design around:

- **Palette**: stone/parchment background, oxblood ink for Sanskrit text,
  brass accents for citation numbers — a manuscript/epigraphic direction,
  deliberately not the generic cream+terracotta or near-black+neon looks
  common in AI-generated UI.
- **Type**: Fraunces (display), Work Sans (UI text), and critically Noto
  Sans Gujarati — a font that actually supports Gujarati glyphs, not a
  decorative substitute.
- **Signature element**: the citation card itself, styled like manuscript
  marginalia (thin brass rule, small chapter.verse badge, Sanskrit set
  apart from the gloss beneath). This ties directly to what the app does
  — citation-grounded answers — rather than being decoration.

## What's validated vs. not

**Tested in this session:**
- `POST /ask` and `GET /health` — ran real HTTP requests via FastAPI's
  TestClient against the real (test) Postgres, with the reasoning
  pipeline's embed/generate calls mocked. Confirmed: correct request
  validation (empty question → 422), correct response shape, correct
  error handling path for `GenerationError`.
- Frontend TypeScript — `tsc --noEmit` passes with zero errors.
- Frontend lint — `eslint` passes with zero warnings.
- Frontend Next.js compilation — Turbopack compiled successfully; static
  page generation succeeded.

**NOT tested — needs your machine:**
- The full production `next build` — this sandbox has no network access
  to `fonts.googleapis.com`, so `next/font/google`'s self-hosting step
  (which needs to download Fraunces/Work Sans/Noto Sans Gujarati at
  build time) fails HERE specifically. This will work normally on any
  machine with real internet access — confirmed the *code* is correct
  via `tsc` and `eslint` independently of the font-fetch step.
- Visual appearance — I designed and wrote the CSS/Tailwind classes
  carefully, but never rendered this in an actual browser (no browser
  tool in this sandbox). Genuinely check how it looks once you build it.
- `docker-compose up` with all three services together — built each
  Dockerfile correctly per documented patterns, but never ran the full
  multi-container stack end-to-end.
- Real question-asking through the actual UI (needs real embeddings from
  Kaggle + a real `HF_API_TOKEN`, neither available in this sandbox).

## How to run the full stack

```bash
cp .env.example .env
# fill in POSTGRES_PASSWORD and HF_API_TOKEN
make up
```

Then open http://localhost:3000. Check `docker compose -f docker/docker-compose.yml logs -f`
if something doesn't come up — this is the first time these three
containers have run together, so expect to debug something real.

## Known gaps / next steps

- No loading skeleton/streaming for the answer — it waits for the full
  response before showing anything. Worth adding if generation latency
  turns out to be slow.
- No conversation memory — each question is independent; the original
  Phase 0 scope didn't specify multi-turn context, worth deciding.
- CORS is currently hardcoded to `http://localhost:3000` — fine for
  local dev, needs revisiting if this ever runs anywhere else.
- No auth on the API — fine for a single-user local demo, would need
  addressing before any public exposure.
