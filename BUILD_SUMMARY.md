# EngageIQ — Build Summary (Final Path)

> A clean, phase-by-phase record of **what was actually built** — the work that makes up the project and the final path taken. For the full journey *including* problems, dead-ends, and course-corrections, see [`PROJECT_LOG.md`](PROJECT_LOG.md). For the forward plan, see [`BUILD_PLAN.md`](BUILD_PLAN.md).
>
> **Status:** living document. **Last updated:** 2026-06-04 (all 6 mandatory capabilities built + verified, 53/53 tests pass; plus a premium UI pass, seamless feed-caching navigation, and an interactive Trends rework; next = hosting + submission artifacts).

---

## What EngageIQ is
A "Smart Engagement Opportunity Scorer" (BAX-423 final project, Option C): a product that ingests engagement opportunities across 15 technical domains, embeds and ranks them for a user, learns from feedback, surfaces trends, and exports a weekly brief — built around an "attention-portfolio / engagement-ROI" idea.

---

## Phase 0 — Foundations & Design
- Studied the assignment and digested all 9 course lectures, mapping each lecture's technique to a project capability.
- Locked the stack: **Python 3.12**, SQLite, **Sentence-BERT + FAISS**, PySpark, Streamlit; sources = **GitHub, Hacker News, Dev.to, Bluesky**.
- Defined the **product experience**: Onboarding → choice-based Calibration → Hub (For You + Trends), captured in a clickable mockup (`mockups/index.html`).
- Defined the **signature approach**: engagement-ROI ranking (impact ÷ effort vs. a weekly time budget), a time-budget portfolio re-ranker, an online weight-learning bandit for interpretable personalization, and a 2-stage retrieval (FAISS KNN → LLM relevance re-rank).

---

## Phase 1 — Data Ingestion & Streaming
- Built the canonical **`Opportunity` schema** (`code/engageiq/schema.py`) — the single structure every source is normalized into — and a pluggable **`SourceAdapter`** interface.
- Built **SQLite storage with two-layer deduplication** (`code/engageiq/storage.py`): a from-scratch **Bloom filter** (Lecture 2) as a fast pre-check + a database primary key as the certain backstop.
- Implemented **four source adapters**, each normalizing to the schema and paginating for volume:
  - **GitHub** — token-authed search for beginner-friendly (`good first issue`) issues.
  - **Hacker News** — recent (2026) stories via Algolia search.
  - **Dev.to** — tagged articles.
  - **Bluesky** — authenticated AT-Protocol search.
- Ran the collector (`code/engageiq/collect.py`, with `--sources/--domains/--limit`) to ingest **21,845 deduplicated, 2026-only records across all 15 technical domains** → `data/engageiq.sqlite`:

  | Source | Records |
  |---|---|
  | Bluesky | 8,572 |
  | Dev.to | 7,469 |
  | Hacker News | 4,638 |
  | GitHub | 1,166 |
  | **Total** | **21,845** |

- Verified domain coverage (all 15 present) and audited dataset quality to guide Phase 2 cleanup.
- *(Added later, 2026-06-02: **Reddit** as a 5th source. Reddit blocks server-side API access, so the public `.json` listings were captured via a browser-console snapshot on reddit.com using the logged-in session — the TA-sanctioned method — then `ingest_reddit.py` normalized **4,826 posts across all 15 domains** into the schema (two capture passes: `top/month`, then `hot` + more subreddits for balance). Then **reclassified by content** (a 25-agent multi-label workflow that also flagged noise), dropping **1,058 noise posts** → Reddit **3,768**, content-verified and multi-label. Current DB: **20,995 records across 5 sources** — Dev.to 6,607 / Bluesky 5,482 / Hacker News 4,138 / Reddit 3,768 / GitHub 1,000. Re-embedded and re-featured; Reddit wired into the personas and scoring.)*

---

## Phase 2 — Embeddings & Retrieval ✅
- Selected the embedding model **`all-MiniLM-L6-v2`** (384-d) via an evidence-based comparison vs `bge-small-en-v1.5` (better domain separation, simpler pipeline).
- Built the **embedder** (`embed.py`), a **FAISS retriever** (`retrieve.py`, exact cosine via `IndexFlatIP`), and a model-agnostic **evaluation suite** (`eval/embedding_eval.py`).
- Encoded all records → **384-d L2-normalized vectors** (cached). Cleaned the data: removed near-duplicates and noise → **19,571 records**.
- Built the **profile→retrieval candidate-generation API** (`Retriever.recommend()`) and encoded the **4 personas** (`personas.py`). Validated end-to-end — each persona retrieves on-target opportunities.
- **Retrieval precision@10 = 0.89.** This is the core matching engine and is robust to domain-label noise.
- Wired a **free LLM provider** (`llm.py`, NVIDIA's hosted OpenAI-compatible API) for the generation tasks — fast model for re-ranking, a stronger reasoning model for Suggested Actions, with a deterministic no-key fallback.
- **Re-classified domains with multi-label Claude agents** (primary + secondaries) and an independent verification gate (**78% verified vs 62% keyword**); **removed 2,344 verified-noise records** → **17,227 clean 2026 opportunities** (≈54% carry multiple domains), embeddings re-aligned.
- Built **cross-encoder + LLM re-rankers** (`rerank.py`) for the Stage-2 step, ready to benchmark against cosine.

---

## Phase 3 — Engagement Scoring & Multi-Stage Ranking ✅
- **Enriched GitHub** (`enrich_github.py`): re-fetched 576 unique repos for **stars, language, contributor count, star-velocity** (cached to `data/github_repo_enrich.json`) — the signals the personas are graded on. Now `community_size`/`language` populated.
- Built the **signal layer** (`features.py`): the spec's 4 signals (relevance, community-health, visibility, effort) plus velocity/recency/standout, each normalized to [0,1] via **per-source percentile (empirical CDF)** so cross-source counts are comparable. Cached to the `opportunity_features` table. Effort estimation in `effort.py` (heuristic + optional LLM).
- Built the **scorer** (`score.py`): **goal-conditioned Engagement-ROI** with **4 engagement modes** (Contribute / Discuss / Monitor / Promote) set by goal×platform — each mode is a weight vector + platform preference + action verb. `impact = Σ weight·signal`; the list ranks by impact, with effort feeding the ROI metric + the knapsack.
- Built the **multi-stage funnel** (`rank.py`): **stratified per-source recall** (FAISS) → hard filters (platform / domain / language-exclusion) → **cross-encoder relevance** (Stage-2) → goal-weighted impact → **MMR diversity** → **time-budget knapsack** ("This Week's Plan", doers) **or velocity radar** (Monitor), plus a **standout-repo lane**. Deterministic **"Why this?"** explanations.
- Built the **evaluation harness** (`eval/ranking_eval.py`, `eval/judge.py`): **LLM-as-judge** relevance (0-3, pooled, **cached** for reproducibility), **NDCG@10 / P@10**, and the **persona pass/fail table**.
- **Results: 12/12 persona pass-criteria PASS** (Sofia/David/Lina/Raj). Pipeline NDCG@10 mean **0.711** (re-validated on the cleaned 5-source data; all 12/12 persona checks still pass, up from 0.664 before Reddit + reclassification). **Matching benchmark** (`eval/retrieval_bench.py`): dense 0.63 · TF-IDF 0.77 · hybrid 0.70 · **cross-encoder 0.76** (wins for skill-match personas) — an honest finding that dense-alone underperforms keyword on this jargon corpus, fixed by the cross-encoder Stage-2. Artifacts in `reports/`.

---

## Phase 6 (in progress) — Product UI (Capability 6)
- **Architecture (revised 2026-06-02):** a first pass built the Hub in Streamlit (`code/app.py`) to prove the engine wiring, but it diverged from the carefully-designed prototype. The team chose to make the **already-iterated prototype the real frontend** (`mockups/prototype.html`) wired to a **FastAPI backend** (`code/api.py`), so the product looks exactly like the design and still serves live engine output.
- **`code/api.py` (FastAPI):** loads the engine once; `GET /api/personas`; `POST /api/recommend` (by `persona_id` or a custom `profile`) returns `recommend()` serialized to JSON (friendly labels, em-dash-stripped copy, mode color, ranked + rail(basket|radar) + standout repos); serves the prototype as the app. Server-side aliases reconcile the prototype's domain/platform keys with the engine's.
- **`mockups/prototype.html` (frontend):** the onboarding (build-your-own form + 4 persona cards + detail modal) and Hub now fetch live recommendations. UI conventions cleaned up to spec: bookmark as a top-right icon, two consistent Engage/Skip buttons, no emoji button labels. Verified in-browser: onboarding → Use persona → real Discuss-mode Hub with live cards, This-week's-plan, and standout repos; console clean.
- *(Superseded but kept for reference: the Streamlit Hub below.)* The earlier Streamlit Hub (`code/app.py`):
- **What renders:** a persona picker (sidebar) → a color-coded **engagement-mode banner** → a **"For you"** feed of real cards (source badge, friendly domain label, the deterministic "why this", ROI / relevance / effort, the mode-specific **suggested action**, and engage / skip / save feedback controls) → a **right rail that swaps by mode**: the time-budget knapsack **"This week's plan"** for doers (Contribute / Discuss / Promote) or the **Velocity Radar** for Monitor, plus a **standout-repo lane**.
- Engine is loaded once (`st.cache_resource`) and recommendations are memoized per persona (`st.cache_data`), so switching personas is instant. Domain keys are mapped to friendly labels (no raw `machine_learning` on screen) and an em-dash stripper keeps all engine-produced copy on house style.
- **Visually verified end-to-end** (Claude-in-Chrome, localhost:8501) across **all 4 personas × 4 modes**: Sofia/Contribute, David/Discuss, Lina/Monitor (radar), Raj/Promote, each showing correct real data, distinct mode color, and the right rail switching appropriately. Reddit flows through in-product, confirming the 5th source is live.
- *(Streamlit Hub superseded; the prototype below is the real product.)*

### Phase 6 continued — the real prototype Hub + ranking upgrades (2026-06-02)
- **Redesign + theming:** the Hub was rebuilt in the prototype with a **light/dark theme system** (CSS variables + a persisted toggle), source **filter chips**, and a planner right rail (UI research; mockups in `mockups/redesign/`).
- **Adaptive learning (Phase 4):** `feedback.py` per-session online weight-learner; engage / skip-with-reason / save → a "learning" toast + a "Re-rank with my feedback" button; `eval/adaptive_eval.py` shows sim NDCG@5 0.68 → 0.96.
- **Contribution-worthiness ranking:** `features.py` adds `repo_reputation` (sweet-spot on stars), `issue_quality` (write-up quality), and `contribution_worth`; `score.py` contribute mode weights it + a multiplicative gate, so established, well-specified repos outrank random/obscure ones (12/12 personas pass; pipeline NDCG@10 0.711 → 0.724). Research: `reports/research_oss_contribution.md`.
- **Cards + detail popup:** **LLM card summaries** (`summarize.py`, 3-4 sentence task descriptions, cached `data/summaries.json`, 195 prewarmed) replace the raw truncation; clicking a card opens a **detail popup** (what it is / why it ranked with plain-language bands / how the time was derived / the repo read). Pagination (10/page) with rank+page telemetry. "Leverage opportunity" opens the real issue + logs it; "Not for me" captures a dismissal reason.
### Phase 6 continued — the plan-driven Hub (2026-06-02)
The product now flows **persona/profile -> a weekly Plan screen -> the Hub**, so the user steers where their attention goes before they ever see a card.
- **New module `code/engageiq/plan.py`:** the four **activity buckets** (Contribute -> GitHub, Learn -> Dev.to, Discuss & watch -> Reddit + Bluesky, News -> Hacker News) and the **recommended weekly split**. The split is a transparent heuristic: a per-mode base (the user's goal gets the majority, with a recommended taste of the others) tuned by an **experience read** (a newbie like Sofia leans more on Learn: 38/37/15/10; an expert like David leans Contribute/Discuss: 29/45/.../6). A prose **"read" of the goal** is LLM-generated and **cached to `data/plan_reads.json`** (all 4 personas prewarmed; deterministic fallback offline). `hot_topics()` surfaces the most active areas from the data.
- **Plan screen (`mockups/prototype.html`, new Screen 2):** the read, an "Active in your space" topic strip, and four **bucket sliders** (logo + share + reason). Sliders **rebalance to always sum 100**, and any bucket can go to **0% (it greys out and drops from the feed)**. A "week at a glance" rail previews the mix as `~4 of 10 cards · ~1.9 hrs` per bucket.
- **Hub, now bucket-driven:** the confirmed split feeds `POST /api/recommend` (`split` param). The engine (`rank.py`) gains a **source override** (active buckets widen/narrow which sources are pulled, e.g. Sofia gets Dev.to "Learn" items her base persona never listed) and a **mild per-bucket prior** on impact (`plan.bucket_weights_from_split`). The Hub chips become **activity buckets with logos + share** (a 0% bucket has no chip); **"For you" is an interleaved mix** (weighted round-robin so the visible feed matches the split), and each chip drills into its bucket. A **"This week's mix"** rail card + an **Adjust plan** button let the user re-tune without redoing onboarding.
- **Temporary "Reset learning" button** + `POST /api/feedback/clear-all` wipe all stored reactions (`feedback_state.json` + `feedback_log.jsonl`) so testing starts fresh; removable later.
- **Verified end-to-end in-browser** (Sofia): plan read + split, News -> 0% drop + rebalance to 42/40/18, Hub chips with no News chip, interleaved feed (GitHub/Dev.to alternating), Learn chip -> 19 Dev.to-only items, Reset learning reloads clean. Console clean throughout. The named-persona ranking eval is unaffected (the new engine params are additive with `None` defaults, so the default path is byte-for-byte the old behaviour; 12/12 still holds).

### Phase 6 continued — per-user persistence + idempotent feedback (2026-06-03)
Moved the user-facing state from ephemeral (a throwaway session id, flat files, append-every-click) to a real per-user database, so actions are idempotent, persistent, and analyzable.
- **New DB `data/engage.sqlite` (`code/engageiq/store.py`),** kept SEPARATE from the read-only corpus so graders still get a clean snapshot. Five tables: **users**, **personas** (each profile + its confirmed split), **interactions** (ONE row per card = the idempotent current disposition), **events** (append-only stream of every click WITH the signal-vector snapshot, for the "why" analysis later), **learner_state** (the RL weights). Thread-safe (`check_same_thread=False` + a lock, since FastAPI serves on a threadpool). Identity = the browser's localStorage id (no login); everything is scoped to **(user, persona)** because the RL weights are mode-specific.
- **Idempotent actions.** Clicking **Leverage** many times is ONE interest: the RL trains and the disposition flips only on the FIRST transition (every click still logs an event for history), and the card then shows a calm **"Interested" state + "Open again"**. **Not for me** opens an **inline reason toggle** (not a popup), stores the reason, and **greys the card with an Undo** (the user's choice). **Save/unsave** persists.
- **RL survives refresh.** `recommend()` loads the learned weights per (user, persona) and keeps adapting, so learning accumulates across reloads. On reload each card's **saved disposition is restored** (interested / dismissed / saved). Verified in-browser: after a full refresh, the engaged card is still "Interested" and the learner reloads at n reactions.
- **Card redesign (Compact + full summary):** the feed card is the tight "compact" framing the user picked, but with the **whole LLM summary** shown (no clamp) and the heavy "why" box moved into the detail popup. Cleaner top row, one domain line, state-aware footer.
- **Summaries gap fixed:** Dev.to/Bluesky cards were missing summaries (the prewarm only covered the base personas' narrow sources). Fixed the prewarm to cover all sources, and a **7-agent Claude Workflow** (`code/dump_for_summary.py` + `merge_summaries.py`) filled the rest with no rate limit. Cache 195 -> 381; Dev.to 4 -> 37, Bluesky 35 -> 138.
- **"Reset learning"** now wipes the whole DB via `POST /api/feedback/clear-all` (kept easy to reset while we iterate). `feedback.Store` (the old JSON store) is superseded by `store.EngageStore`; `feedback.Learner` (the LMS math) is reused.

### Phase 6 continued — multi-profile gallery + the feedback loop made visible (2026-06-03)
Closed two gaps the user named while reviewing the flow ("I can't show multiple profiles to the professor"; "a refresh doesn't change the list, is the feedback even implemented?").
- **Multi-profile gallery.** Built profiles persist to `localStorage["eiq-profiles"]` and show as a **"Your profiles"** card grid on onboarding (avatar, name, goal, domain tags, **Use / Edit / Delete**), above "+ Build a new profile" and the example personas. You can create many, compare them side by side, and switch between them (the Hub topbar **"Switch"** returns to the gallery). A stable `EDIT_KEY` makes Edit update in place and Build fork a new key; `routeTo` rebuilds a custom profile from the gallery on a direct `/hub/<key>` load; Delete also calls **`POST /api/persona/delete`** -> `store.delete_persona` (removes that persona's interactions / events / learner_state / personas rows).
- **Feedback visibly re-ranks.** Dismissed cards now **leave the feed** on reload (`api.recommend` filters stored-`dismissed` items via `_keep`), so a refresh reflects your reactions. They collect in the Activity **Passed** section, now a proper row UI (logo + title + domain + reason chip + a **Restore to feed** button that undoes the dismissal and refreshes in place). A **"Your feed adapted to your reactions"** banner shows on reload; the learner is a bit more responsive (lr 0.20 -> 0.30); and the learned-summary copy was cleaned up (correct singular/plural, and reads as "leaning into: ... ; easing off: ..." instead of the old double-prepositioned "respond more to matches your interests").
- **Verified end to end in Chrome:** 2 profiles created -> both in the gallery -> Use -> plan -> Hub (60 cards) -> dismiss one -> reload shows 59 with the card gone + the banner -> Activity Passed shows it + Restore -> Restore -> feed back to 60. Console clean. The named-persona ranking path is untouched (the gallery + `_keep` only affect custom profiles / sessions with stored dismissals), so the 12/12 persona eval still holds.

### Phase 6 continued — detail page, concepts, Rising fix, Nemotron summarization, UX run (2026-06-04)
A dense polish pass plus a summarization cost pivot, all verified in Chrome.
- **Opportunity detail PAGE** at `/opportunity/<id>` replaces the old popup: a real, shareable, refresh-safe page (source / type / action chip / title / actions, the synopsis, an **action-aware Concepts section**, why-it-fits, full body, repo, effort). Concepts are generated **on-demand** the first time a card is opened (`POST /api/opportunity`, NVIDIA `llm.chat`, cached to `data/concepts.json`), so only opened cards cost a call.
- **Per-card bucket chip** (Contribute / Learn / Discuss / News) showing the intended action, colored to match the rail.
- **"Rising" badge fixed:** it used a per-source velocity percentile, so 0-engagement Bluesky posts were flagged "Rising." Now gated on real engagement (`num_comments + score` over a per-source floor) so only genuinely active posts qualify.
- **Learning made central:** removed the easy-to-miss floating toast; the engagement mode and the live "what EngageIQ learned" line are now one **mode card** at the top, updating on every reaction.
- **Two flow fixes:** re-entering a configured persona now goes **straight to the Hub** (the Plan is only shown the first time); new profiles get **readable URLs** (`/hub/saurav-...`).
- **Smaller fixes:** Reddit added to the profile form (all 5 sources selectable); a **"Wiggle room %"** row so "This week's mix" reconciles to 100; a **no-cache** header on the app shell (fixes browsers serving a stale app); a **boot warm-up** so the first feed load is ~0.3s instead of ~8s.
- **Summarization moved to Nemotron (cost):** the Claude sub-agent passes covered ~9.4k items but cost ~10M tokens. Switched to a free, resumable background `nemotron_summarize.py` (NVIDIA Nemotron 49B) for the rest, at zero Claude tokens. Running now (~49% and climbing).

### Phase 5 — Batch Analytics & Trend Detection (2026-06-04)
A single batch pass over the **full 20,995-record corpus** that the dashboard scopes to each persona.
- **Probabilistic sketches from scratch (Lecture 2)** — `code/sketches.py`: a **Count-Min Sketch** (term frequencies in fixed memory, never-undercount guarantee) and a **HyperLogLog** (distinct counts in 16 KB). Benchmarked against the exact answer on the real streams: **CMS 1.16 % max error at 9.8× less memory**, **HLL 0.43 % error at 48.7× smaller**. Both *merge* without rescanning, which is what lets the dashboard roll a persona's domains up on the fly.
- **The batch job** — `code/analytics.py` (pandas, ~8 s) writes `data/trends.json`, `data/trends_coords.npz`, and `reports/phase5_analytics.txt`. Per domain: category distribution, trending terms (CMS), most active communities (engagement + HLL distinct authors), weekly volume, **week-over-week share momentum**, a rising score, related domains (cosine similarity of MiniLM centroids), and a t-SNE topic-map sample. The same aggregates are also written as a real **PySpark** job (`code/analytics_spark.py`, Lecture 4) for a cluster with a JVM.
  - *Honest limitation, surfaced in the brief:* the offline snapshot skews recent (a collection artifact), so trend signals use each domain's **share of weekly activity on complete weeks**, which is robust to that bias.
- **Persona-scoped trends** — `POST /api/trends` merges the per-domain sketches (HLL registers by max for distinct reach, term counts by sum) to produce KPIs, momentum, communities, a weekly share series, related domains, and breakout opportunities for *that* persona; `code/trends_render.py` draws a persona-highlighted topic map from the saved coords (cached).
- **The Trends UI** — a **tab shell** (Opportunities / Trends / Activity) and a persona-specific **Trends overview** at `/trends/<persona>`: KPI tiles, then a **2x2**: (top-left) an **interactive topic mind map** — a from-scratch force-directed node graph (your domains sized by volume, adjacent domains, similarity edges) that opens to an enlarged popup where you drag nodes, hover for stats, **click a domain to fan out its trending terms**, and double-click to drill into its feed; (top-right) **Momentum** with the metric defined on the card (a domain's share of weekly activity, latest complete week vs the one before, in pp); (bottom-left) **Fresh this week** (newest opportunities by source over the last 7 days of the snapshot + the freshest clickable items); (bottom-right) **Most active communities** (hyperlinked out to reddit/HN/dev.to/bsky/github). Clicking a domain (chip or map node) **drills** into the Opportunities feed filtered to that domain. *(Iterated across 2026-06-04: static topic-map PNG → interactive over-time chart → this 2x2 with a real node mind map; see the changelog.)*

### Phase 6 — Dashboard, "Why this?", Suggested Actions & the Engagement Brief (2026-06-04)
- The dashboard (ranked opportunities + scores + **"Why this?"** + feedback controls) was already in place; this phase added the two missing pieces.
- **LLM Suggested Actions** — `_action_for` drafts a concrete, source-appropriate engagement action (a suggested contribution, comment, reply, or PR) plus a rationale; shown as a green **"Suggested … [AI]"** box with Copy-draft on every opportunity's detail page.
- **The downloadable engagement brief** — `POST /api/brief/html` renders a **self-contained, print-ready** document (the persona's weekly plan, top opportunities with Why-this + a suggested action, the trends in their space, and what the learner has learned). The in-app `/brief/<persona>` screen shows it in an iframe with **Save as PDF** and **Download brief**.
- **Tests** — `code/test_capabilities_5_6.py`: **53 checks, all passing** (sketch accuracy + merge property, trends.json/coords schema + benchmark bounds, and the trends/brief endpoints).

### The 6 mandatory capabilities — coverage
| # | Capability | Status | Where |
|---|---|---|---|
| 1 | Multi-source ingestion + streaming + dedup + storage | ✅ | `engageiq/collect.py`, Bloom-filter dedup, `data/engageiq.sqlite` (20,995 records, 5 sources) |
| 2 | Embeddings + similarity retrieval (Sentence-BERT + ANN) | ✅ | `embed.py` / `retrieve.py` (MiniLM 384-d, FAISS, P@10 0.89) |
| 3 | Engagement scoring + multi-stage ranking + a metric | ✅ | `features.py` / `score.py` / `rank.py` (cross-encoder + MMR; NDCG@10 0.711, 12/12 personas) |
| 4 | Adaptive learning from feedback (50+ rounds) | ✅ | `feedback.py` + `store.py` (online bandit; NDCG@5 0.679 → 0.960 over 60 rounds) |
| 5 | **Batch analytics & trend detection** | ✅ | `sketches.py` / `analytics.py` / `analytics_spark.py` / `/api/trends` / Trends tab |
| 6 | **Dashboard + "Why this?" + LLM suggested actions + downloadable brief** | ✅ | Hub + detail page (`suggested_action`) + `/api/brief/html` + `/brief/<persona>` |

### Accounts, DB-backed profiles, and a first-run tour (2026-06-04)
A simple identity layer so profiles persist in the DB and reload by name, plus a polished onboarding.
- **Name-based accounts (no password).** `POST /api/login {name}` finds-or-creates an account (`store.find_or_create_user`, case-insensitive) and returns a `user_id` that becomes the session id scoping everything. Identity only, not security (no passwords, which also fits the safety rules; no cross-device sync without real auth).
- **Profiles in the DB.** A new `profiles` table holds each account's custom profiles; `POST /api/profiles` (upsert), `/api/profiles/list`, and `/api/profiles/delete` back the gallery, which now loads from the server instead of `localStorage`. Each account sees the 4 shared example personas plus its own private profiles, with learning kept per account.
- **First-run tour.** A three-step animated walkthrough (the sources; a ranked card with "Why this" and a suggested action; trends and the brief) shown once per browser, then a name-only login. An account chip with Log out sits on the gallery; the intro is re-watchable.
- **Tested:** `code/test_accounts.py` (15 checks, all passing) plus a full browser walkthrough (tour → login → build → log out → log back in with the profile restored → a second account isolated).

### Premium UI pass, seamless navigation, and interactive Trends (2026-06-04)
A polish pass to make the product feel buyable, with every frontend change verified by measuring computed values in-browser (not eyeballing).
- **Premium visual system.** A layered design pass at the end of the stylesheet: elevation/glow tokens, spring/ease motion curves, an ambient body mesh-gradient, a glass sticky topbar (`backdrop-filter`), button sheen + spring-press, card hover-lift, an aurora-animated onboarding/login backdrop with frosted-glass cards, and a custom scrollbar. Verified premium across every screen in both light and dark.
- **Seamless navigation (the feed stops reloading when it should not).** A `HUB_LOADED_KEY` guard + `showHub()` restore the already-rendered feed instead of re-fetching, wired into routing, the detail back-button, the tab bar, and a new "Back to feed" button on the build/edit screen. The feed now re-fetches **only** on flows that need it (new persona, confirm plan, reset-learning). Measured: back-from-detail, edit-and-back, and tab switches all kept `/api/recommend` at a single call (zero refetches). Also fixed a bug where re-using a saved custom profile dropped its confirmed split and wrongly routed to the Plan screen instead of the feed.
- **Interactive Trends rework.** The static topic-map image and the Count-Min term-bar panel were retired; the week-over-week share chart moved into the hero and became interactive (per-week hit columns drive a crosshair, per-series dots, and a frosted popup of each domain's share; latest week previewed on load). "Most active communities" is widened to a full-width grid with every community hyperlinked out to its source (reddit/HN/dev.to/bsky/github). Measured: term-bars and map gone, 6 community links resolve to the right hosts, 22 hover columns, tooltip tracks the correct week and stays in bounds.

- **NEXT (still required for the grade):** **hosting** (a live public URL, 20 pts) and the **submission artifacts** — the ≤4-page technical `brief.pdf` (architecture, BAX-423 technique choices, persona pass/fail table, limitations), `prompts.md`, and the `README`. Forward plan in `BUILD_PLAN.md`.
