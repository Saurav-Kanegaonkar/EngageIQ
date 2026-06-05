# EngageIQ — Build Plan

> **Status:** living draft (v1). We go **phase by phase**; you confirm a phase before I build it.
> **Timeline:** intentionally removed as a constraint. The goal is the most complete, coherent, defensible project we can build.
>
> **Current status (2026-06-02):** Phases 0-3 DONE — the engine is complete. Data pipeline is now **5 sources / 20,995 content-verified records** (Reddit added via a browser-console snapshot, then reclassified by content + noise-dropped). **12/12 persona pass-criteria passing.** UI: the prototype is now the **real frontend**, wired to a **FastAPI backend** (`code/api.py`); onboarding → Hub renders live engine recommendations (verified in-browser). **Done since:** Phase 4 adaptive learning (online weight-learner; sim NDCG 0.68→0.96), a full Hub redesign with light/dark themes, pagination + rank/page telemetry, a per-card **detail popup** (what it is / why it ranked / how the time was derived / repo read), a research-driven **contribution-worthiness ranking signal** (down-ranks random/obscure repos; 12/12 personas pass, NDCG 0.711→0.724), and **LLM card summaries** (3-4 sentence task descriptions, cached in `data/summaries.json`, 195 prewarmed).

> ### THE NEXT BUILD: the plan-driven Hub (build as ONE coherent flow, NOT piecemeal quick-wins)
> 1. **Plan screen (new Screen 2, between persona and Hub).** On persona pick/build, generate: (a) an LLM read of what the user wants, (b) a recommended **weekly activity split** across the buckets, mostly their stated goal PLUS a recommended bit of the others, with the reasoning shown, (c) this week's hot topics. The user can **adjust the percentages, including to 0% (a 0% bucket disappears)**. Percentages: a transparent heuristic by experience level (a newbie like Sofia leans learn/watch; an expert like David leans contribute/discuss) with an LLM narrative explaining it. A **weekly synopsis, never a day-by-day schedule.**
> 2. **Activity buckets = the chips.** Contribute (GitHub) · Learn (Dev.to) · Discuss & watch (Reddit + Bluesky) · News (Hacker News) · **Saved**. Each chip shows its logo + its share. Classify every opportunity into a bucket from source+type. The confirmed %s drive the **"For you" feed as an interleaved mix** (40% contribute → ~4 of every 10 cards); each chip drills into just that bucket.
> 3. **Summaries.** Keep the structured 3-4 sentence LLM summaries (cached). `dump_for_summary.py` + a sub-agent Workflow can regenerate them higher-quality (Claude agents, no NVIDIA rate limit) if wanted.
> 4. **Wire the confirmed split into ranking** (a small per-bucket weight layered on the worthiness ranking).
>
> **After that:** batch analytics + a Trends tab (Capability 5); a real **database** for persona persistence + week-on-week growth (DEFERRED by the user, needs schema design); hosting (Render / HF Spaces / GCP credits); the technical brief + demo. *(Framework note: pivoted from Streamlit to prototype+FastAPI on 2026-06-02 for design fidelity, the user had heavily iterated the prototype and Streamlit could not match it.)*
> **How to use this doc:** read through, scribble questions anywhere, and react especially to the **[Open Questions](#open-questions-lets-decide-together)** section. Each phase has a **temporary vs permanent** breakdown so you know what's a locked foundation and what will deliberately evolve.

---

## Table of contents
1. [What we're building (north star)](#1-what-were-building-north-star)
2. [What makes a *great* build vs a checkbox build](#2-what-makes-a-great-build-vs-a-checkbox-build)
3. [What makes *our* build distinctive (the signature)](#3-what-makes-our-build-distinctive-the-signature)
4. [Legend: permanent / provisional / deferred](#4-legend)
5. [Technique → lecture map (all 9 lectures)](#5-technique--lecture-map)
6. [The 6 mandatory capabilities](#6-the-6-mandatory-capabilities)
7. [Data sources](#7-data-sources)
8. [Phases at a glance](#8-phases-at-a-glance)
9. [Phase-by-phase detail](#9-phase-by-phase-detail)
10. [Open questions — let's decide together](#open-questions-lets-decide-together)
11. [Submission deliverables checklist](#11-submission-deliverables-checklist)
12. [Phase confirmation log](#12-phase-confirmation-log)

---

## 1. What we're building (north star)

EngageIQ is a **recommender system for an unusual item: units of your professional attention.** It ingests "engagement opportunities" (a GitHub issue to fix, a Reddit thread to join, an HN story to discuss, a Bluesky/Dev.to post to reply to), dedups and stores them, embeds the text, retrieves and ranks the best ones for a given person, learns from their feedback, surfaces trends, and exports a weekly brief — all in a hosted dashboard a non-data-scientist can use.

The assignment is designed so the production-ML pipeline (**ingest → dedup → embed → retrieve → score → rank → learn → serve → analyze**) gives every lecture a natural home.

**Hard constraints (from the spec):** ≥10,000 records across all **15 technical domains**; offline snapshot ≥5,000 so graders run with no API keys; integrate ≥2 BAX-423 techniques from different lectures and **benchmark** them; test on 4 named personas + hidden personas; live hosted URL; 10-min demo; code that doesn't run scores 0 on rubric dims 1–4. **Rubric:** Data Pipeline 15 · Matching/Ranking 20 · Adaptive Learning & Techniques 15 · Hosting 20 · Brief & Demo 30.

---

## 2. What makes a *great* build vs a checkbox build

Hitting the 6 capabilities is the easy part — most submissions will. Three things separate a great project, and they're where we'll spend real effort:

1. **Define "engagement value" honestly.** The APIs give us stars, upvotes, comments, timestamps. Turning those into *defensible proxies* for **relevance, community health, visibility potential, and effort** is the intellectual core. We want principled signals, not an arbitrary weighted sum.
2. **Solve the no-labels problem honestly (the deep one).** We have **no real engagement history.** So "is our ranking good?" and "did learning improve it?" have no ground truth out of the box. We must *construct* relevance judgments per persona and *simulate* a user whose true preferences are **hidden from the learner** — otherwise the benchmarks become circular (we'd grade the system against its own assumptions). This is ~35 of 100 points. See **[Phase 3](#phase-3--relevance-judgments--evaluation-harness)**.
3. **Coherence.** Six capabilities that *reinforce* each other, not six bolted-on modules: streaming velocity feeds the ranking; sketches power the trends view; feedback moves the weights you can *see* in "Why this?".

---

## 3. What makes *our* build distinctive (the signature)

We don't bolt "novelty" on as a separate module — we make the *natural* way we do things distinctive:

- **Ideology — an "attention-portfolio / engagement-ROI" lens.** We rank by **expected impact ÷ estimated effort** against each persona's **weekly time budget** (which the rubric literally grades). The product answers *"given your hours this week, what's the best basket of things to engage with?"* — not just "here's a list."
- **Algorithm — a time-budget knapsack re-ranker** (beyond plain MMR diversity) that returns an optimal *basket* under a time budget, and an **online weight-learning bandit** that personalizes the score while keeping "Why this?" fully interpretable.
- **UI — an impact-vs-effort "efficient frontier"** view + a **"This Week's Plan"**, plus an **in-app A/B benchmark panel** that shows technique impact (NDCG) live.
- **Streaming with a real job** — powering the live **"rising/velocity"** trend signal, not Kafka-for-show.

---

## 4. Legend

| Marker | Meaning |
|---|---|
| 🟢 **Permanent** | A locked foundation. Other phases depend on it, so we design it carefully now and avoid changing it. |
| 🟡 **Provisional** | Built now, but **expected to change** as data quality or a later phase's outcome teaches us what it really needs. |
| 🔴 **Deferred** | Can't be built yet — it **depends on a later phase** existing first. |

---

## 5. Technique → lecture map

All six technique families have a **natural** home here (none forced), and we touch all 9 lectures:

| Family | Lecture | Its natural job in EngageIQ |
|---|---|---|
| **Streaming** | L3 | Ingestion pipeline + live velocity/"rising" trend detection |
| **Sketching** | L2 | Bloom-filter dedup · Count-Min trending topics · HyperLogLog unique authors/repos |
| **Embeddings** | L5 | Semantic retrieval — the heart of matching |
| **Recommendation** | L6 | Learned DCN scorer, *benchmarked against* the interpretable linear scorer |
| **Ranking** | L7 | Multi-stage funnel · NDCG eval · MMR diversity · portfolio knapsack |
| **Reinforcement learning** | L8 / L9 | Adaptive bandit on scoring weights (L8); DQN multi-step engagement sequencing (L9, bonus) |
| *(supporting)* **Parallelism / LLM / prod-ML** | L1 | Async multi-source ingestion · LLM "Suggested Actions" · in-app A/B harness |
| *(supporting)* **PySpark** | L4 | Batch analytics & trend detection over the full dataset |

**Benchmarked (the rubric's "≥2, different lectures"):** (a) embeddings (L5) vs TF-IDF keyword baseline on NDCG@10; (b) adaptive bandit (L8) quality before vs after 50+ feedback rounds. Optional 3rd: linear scorer vs DCN (L6).

---

## 6. The 6 mandatory capabilities

| # | Capability | Delivered in |
|---|---|---|
| 1 | Multi-source ingestion & streaming (+ dedup, structured storage) | Phase 1 |
| 2 | Content embedding & similarity retrieval | Phase 2 |
| 3 | Engagement scoring & multi-stage ranking (report a ranking metric) | Phase 4 (eval'd via Phase 3) |
| 4 | Adaptive learning from feedback (measurable over 50+ rounds) | Phase 5 |
| 5 | Batch analytics & trend detection | Phase 6 |
| 6 | Dashboard + "Why this?" + Suggested Actions + feedback + trends + downloadable brief | Phase 7 |

---

## 7. Data sources

| Source | Auth | Role |
|---|---|---|
| GitHub REST API | token (free) | Repos, issues, "good first issue" tags — primary |
| GH Archive | none | Bulk events + week-over-week velocity/trends |
| Hacker News API | none | Stories + discussion threads |
| **Reddit** ✅ | browser-console `.json` snapshot (official API + Devvit both gated; the TA-sanctioned method) | Discussion threads; **3,768 content-reclassified posts** (Sofia/David/Raj/Lina platforms) |
| **Bluesky** | none | Modern social tech engagement (Twitter replacement) |
| **Dev.to** | none | Tech blogging; tags map ~1:1 to the 15 domains |

Twitter/X **dropped** — no free tier in 2026 (pay-per-read), breaks grader reproducibility. Substack (RSS only) = stretch.

---

## 8. Phases at a glance

| # | Phase | Capability | Lecture concepts | Mostly… |
|---|---|---|---|---|
| 0 | Foundation & Scaffold | — | — | 🟢 Permanent |
| 1 | Multi-Source Ingestion & Streaming | Cap 1 | L3, L2, L1 | 🟢 + 🟡 skin |
| 2 | Content Embedding & Retrieval | Cap 2 | L5 | 🟢 + 🟡 |
| 3 | Relevance Judgments & Evaluation Harness | (enables 3,4 eval) | L7 metrics, L8 sim | 🟡 |
| 4 | Engagement Scoring & Multi-Stage Ranking | Cap 3 | L7, L6, +ROI/knapsack | 🟢 arch / 🟡 signals |
| 5 | Adaptive Learning from Feedback | Cap 4 | L8 (L9 bonus) | 🟢 arch / 🟡 tuning |
| 6 | Batch Analytics & Trend Detection | Cap 5 | L4, L2 | 🟢 + 🟡 |
| 7 | Dashboard & Engagement Brief | Cap 6 | L1 | 🟡 (UX iterates) |
| 8 | Hosting & Deployment | — | — | 🟢 |
| 9 | Technical Brief, Persona Pass/Fail & Demo | — | — | 🟢 (final outputs) |

**Dependency flow:** `0 → 1 → 2 → 3 → 4 → 5`; `1 → 6`; `{2,4,5,6} → 7 → 8`; `everything → 9`.

---

## 9. Phase-by-phase detail

### Phase 0 — Foundation & Scaffold
- **Goal:** the skeleton everything hangs off; get the contracts right early.
- **What gets built:** repo layout (`code/`, `data/`, `notebooks/`, `brief/`); Python env + `requirements.txt`; `.env.example` (gitignored secrets); config loader; the **canonical `Opportunity` schema** (the one common record shape all sources convert into); the **15-domain taxonomy**; the **4 personas encoded** as structured profiles (interests, goal, platforms, weekly time budget — given by spec); SQLite storage layer; `git init`.
- **Lecture concepts:** none directly (foundation).
- **Dependencies:** none.
- **Temp vs permanent:**
  | Item | Status | Why |
  |---|---|---|
  | Canonical `Opportunity` schema (core fields) | 🟢 | Contract everything depends on. |
  | Persona profile configs | 🟢 | Given by the spec; fixed. |
  | Storage layer (SQLite) | 🟢 | Portable snapshot format graders use. |
  | 15-domain query configs | 🟡 | Tuned for coverage/quality once we see real data. |
- **Done when:** `pip install` works, schema + personas + storage import cleanly.

### Phase 1 — Multi-Source Ingestion & Streaming  *(Capability 1)*
- **Goal:** a real, demonstrable pipeline pulling all 6 sources across 15 domains → deduped, structured records → the ≥10k snapshot.
- **What gets built:** a pluggable `SourceAdapter` interface + 6 adapters; each adapter **normalizes** its source's raw JSON into the canonical schema; a **Kafka** producer/consumer path (pollers → `raw_opportunities` topic → consumer → dedup → store; at-least-once + idempotent consumer); **Bloom-filter** exact dedup; SQLite store + a Parquet export hook for Phase 6; the full scrape.
- **Lecture concepts:** **L3** streaming/Kafka (core) · **L2** Bloom filter (dedup) · **L1** async `asyncio.gather` (poll 6 I/O-bound sources concurrently).
- **Dependencies:** Phase 0 (schema, storage). GitHub + Reddit need credentials; the other 4 don't.
- **Temp vs permanent:**
  | Item | Status | Why |
  |---|---|---|
  | `SourceAdapter` interface, Kafka architecture, Bloom dedup | 🟢 | Architectural bedrock. |
  | Each adapter's field mapping (core fields) | 🟢 | Stable once schema is locked. |
  | Per-domain query configs | 🟡 | Tune for coverage/balance per domain. |
  | Exact engagement-signal fields extracted | 🟡 | **Extend once Phase 4 scoring reveals what signals it needs.** |
  | Velocity/"rising" raw counters | 🟡 | Refined alongside Phase 6 analytics. |
  | Semantic near-duplicate dedup | 🔴 | **Needs Phase 2 embeddings.** |
  | The scraped dataset | 🟡 → 🟢 at the end | Top-up/rebalance until we freeze a final snapshot. |
- **Done when (Checkpoint 1):** 10k+ records, all 15 domains present, dedup verified, snapshot built.

### Phase 2 — Content Embedding & Similarity Retrieval  *(Capability 2)*
- **Goal:** semantic retrieval of opportunities for a person.
- **What gets built:** text prep (title+body, cleaned); **Sentence-BERT** (`all-MiniLM-L6-v2`, 384-d) encoding of all opportunities; **FAISS** index; persona/user profile → query vector; ANN top-K retrieval = candidate generation; cached embeddings on disk. (Also unlocks semantic near-dup dedup back in Phase 1.)
- **Lecture concepts:** **L5** embeddings + FAISS.
- **Dependencies:** Phase 1 (records to embed).
- **Temp vs permanent:**
  | Item | Status | Why |
  |---|---|---|
  | Embedding pipeline + FAISS architecture | 🟢 | Stable core of matching. |
  | Model choice (`all-MiniLM-L6-v2`) | 🟢 (could upgrade) | Solid default; swappable. |
  | Index type (Flat vs IVF) | 🟡 | Depends on final data size. |
  | Profile→vector method | 🟡 | Refine how a persona becomes a query. |
  | Cached embedding files | 🟡 | Regenerate when data/schema changes. |
- **Done when:** given a profile, FAISS returns sensible top-K in <1s.

### Phase 3 — Relevance Judgments & Evaluation Harness
- **Goal:** *how we know if we're good* — the honest answer to the no-labels problem. (Not a graded capability by itself, but it's what makes Capabilities 3 & 4 credible.)
- **What gets built:** (a) **relevance judgments** — for each persona, a constructed ground truth of which opportunities *should* rank high, derived **independently of the scorer's own features** (e.g., domain match + persona pass-criteria + light LLM-as-judge) so evaluation isn't circular; (b) ranking-metric code (**NDCG@k, Precision@k, Recall@k** — from the L7 lab); (c) a **user simulator** — a synthetic persona whose true preferences are **hidden from the learner**, used in Phase 5 to generate honest feedback.
- **Lecture concepts:** **L7** ranking metrics · **L8** (the simulator is the RL "environment").
- **Dependencies:** Phase 0 personas, Phase 2 retrieval.
- **Temp vs permanent:**
  | Item | Status | Why |
  |---|---|---|
  | Metric code (NDCG/P@k/Recall) | 🟢 | Standard, stable. |
  | Relevance-judgment construction | 🟡 | Refine the labeling so it's fair and non-circular. |
  | User-simulator design | 🟡 | Tune hidden-preference model so learning is demonstrable yet honest. |
- **Done when:** we can score any ranking with NDCG@10 against per-persona judgments, and the simulator emits engage/skip/bookmark events.

### Phase 4 — Engagement Scoring & Multi-Stage Ranking  *(Capability 3)*
- **Goal:** a defensible ranked ordering + a reported metric.
- **What gets built:** per-opportunity **signals** (relevance, community health, visibility potential, effort); an **interpretable composite scorer** (weighted → exact "Why this?"); the **ROI score** (impact ÷ effort); the multi-stage funnel (FAISS candidates → score → **MMR** diversity → **time-budget knapsack** basket); **NDCG@10** reported; **Benchmark #1** (embeddings vs TF-IDF). *Optional:* a **DCN** learned scorer (L6) as a benchmarked alternative.
- **Lecture concepts:** **L7** multi-stage + NDCG + MMR · **L6** DCN (optional) · + ROI/knapsack signature.
- **Dependencies:** Phases 2 (candidates) + 3 (eval).
- **Temp vs permanent:**
  | Item | Status | Why |
  |---|---|---|
  | Multi-stage architecture (candidate→score→rerank) | 🟢 | Stable pipeline shape. |
  | Signal definitions (relevance/health/visibility/effort) | 🟡 | Refine proxies as we validate them on real data. |
  | The exact scoring weights | 🟡 | **Learned/tuned in Phase 5.** |
  | ROI + knapsack formulation | 🟡 | Refine impact/effort estimation. |
  | DCN scorer | 🟡 / optional | Benchmark experiment; keep only if it earns its place. |
- **Done when (Checkpoint 2):** ranked list with NDCG@10; embedding-vs-keyword benchmark table.

### Phase 5 — Adaptive Learning from Feedback  *(Capability 4)*
- **Goal:** recommendations measurably improve from feedback.
- **What gets built:** an **online weight-learning contextual bandit** (Thompson sampling, L8) that updates the *scoring weights* from engage/skip/bookmark — keeps "Why this?" interpretable; a **50+ round simulation** (driven by Phase 3's hidden-preference simulator) showing rising NDCG/reward; **Benchmark #2** (before vs after). *Optional bonus:* **DQN** (L9) for multi-step engagement sequencing (star → issue → PR).
- **Lecture concepts:** **L8** bandit/Thompson · **L9** DQN (optional bonus).
- **Dependencies:** Phases 3 (simulator) + 4 (the weights it tunes).
- **Temp vs permanent:**
  | Item | Status | Why |
  |---|---|---|
  | Bandit/Thompson mechanism | 🟢 | Architecture is stable. |
  | Reward shaping + learning rate | 🟡 | Tune for honest, demonstrable improvement. |
  | DQN sequencing | 🔴 / optional | Bonus; only if base learning is solid first. |
- **Done when (Checkpoint 3):** chart of measurable improvement over 50+ rounds → "≥2 benchmarked techniques" satisfied.

### Phase 6 — Batch Analytics & Trend Detection  *(Capability 5)*
- **Goal:** aggregate insight over the full dataset.
- **What gets built:** a **PySpark** batch job (trending topics, most active communities, engagement volume over time, domain distributions, week-over-week velocity); **Count-Min Sketch + HyperLogLog** (L2) for streaming trend frequency + unique authors/repos; analytics tables for the dashboard.
- **Lecture concepts:** **L4** PySpark · **L2** CMS/HLL.
- **Dependencies:** Phase 1 (data; runs in parallel with 2–5).
- **Temp vs permanent:**
  | Item | Status | Why |
  |---|---|---|
  | PySpark job structure + sketches | 🟢 | Stable. |
  | Which specific aggregates/trends we surface | 🟡 | Driven by what's interesting in the data + dashboard needs. |
- **Done when:** analytics tables generated; trend numbers sane.

### Phase 7 — Dashboard & Engagement Brief  *(Capability 6)*
- **Goal:** the product that ties everything together.
- **What gets built:** Streamlit app — profile/persona selector; ranked opportunities + scores; **deterministic "Why this?"**; **"Suggested Actions"** (template + optional LLM, offline cache); feedback controls (wired to Phase 5); **trend visualizations** (Phase 6); **signature UI** (impact-vs-effort efficient frontier + "This Week's Plan"); **in-app A/B benchmark panel**; **downloadable weekly brief** (PDF/CSV).
- **Locked onboarding design (prototyped 2026-06-02, `mockups/prototype.html`):** the landing leads with a compact **"Build your own profile"** entry, then **4 persona cards** (photo, role, mode badge, 2-line description); clicking a card opens a **detail modal** (full profile: goal, focus-area domains, per-platform Observe/Engage, time budget, skills) with **Use / Edit & customize**. The form has 4 numbered sections: (1) **About you** = photo upload + name + a rich "goal & about you" box; (2) **Topics** = the 15 domains as a 3×5 chip grid; (3) **Where you'll engage, and how** = the 4 platforms as a 2×2 grid, each with an **Observe / Engage** intent (key product idea: intent is *per-platform* — Observe → monitor/radar ranking, Engage → action ranking); (4) **Limits** = time budget + "anything to avoid". Framework = **Streamlit (styled)**; Gemini reserved for the eval judge + Suggested Actions (cached so graders run offline).
- **Locked Hub design:** mode banner (color-coded) → **For You** cards (source badge, Why-this, Suggested action, engage/skip/bookmark) → a right rail that swaps by mode: **This Week's Plan** (time-budget knapsack basket) for doers, **velocity radar** for monitors, plus a **standout-repos lane** and an **impact-vs-effort** scatter; a Trends tab (Phase 6) and the downloadable brief.
- **Lecture concepts:** **L1** LLM "Suggested Actions" + A/B framing.
- **Dependencies:** Phases 2, 4, 5, 6.
- **Temp vs permanent:**
  | Item | Status | Why |
  |---|---|---|
  | Streamlit app skeleton | 🟢 | Stable host for the UI. |
  | Specific views, layout, charts | 🟡 | UX iterates with your feedback. |
  | "Why this?" logic + Suggested Actions | 🟡 | Depends on final scoring components. |
- **Done when:** all 6 capabilities visible and usable in one app.

### Phase 8 — Hosting & Deployment
- **Goal:** the live public URL (20 pts).
- **What gets built:** public GitHub repo; deploy to **Streamlit Community Cloud**; bundle the offline snapshot so it runs keyless; Streamlit secrets; smoke-test the live URL.
- **Dependencies:** Phase 7.
- **Temp vs permanent:** 🟢 once configured. (You: create the repo + connect Streamlit.)
- **Done when (Checkpoint 4):** the app is live and works for a fresh visitor.

### Phase 9 — Technical Brief, Persona Pass/Fail & Demo  *(30 pts)*
- **Goal:** the deliverables that carry the most weight.
- **What gets built:** run all 4 personas + hidden-persona-style probes → **pass/fail table**; the **≤4-page technical brief** (architecture, technique choices + rationale + benchmarks, persona results, limitations); **`prompts.md`**; **README** (single-command run); rehearsed **10-min demo**.
- **Dependencies:** everything.
- **Temp vs permanent:** 🟢 final outputs (brief written last so it reflects the finished system).
- **Done when (Checkpoint 5):** ZIP-ready submission + rehearsed demo.

---

## Open questions — let's decide together

These shape the build; I have a lean on each but want your call.

1. **No-labels / relevance judgments (most important).** How do we construct "ground truth" for ranking without it being circular? My lean: derive judgments from *persona pass-criteria + domain match + light LLM-as-judge*, kept separate from the scorer's own features. The user simulator gets *hidden* preferences. Agree with this direction?
2. **Data volume.** Floor is 10k. I lean toward **~25–50k** for a richer retrieval pool and more real trends. How big do you want to go?
3. **Streaming's real job.** Confirm streaming earns its place via the live **velocity/"rising"** signal (vs. just box-checking Kafka).
4. **Semantic near-dup dedup.** Worth doing (catches the same story cross-posted to HN/Reddit/Bluesky)? It's deferred until Phase 2 either way.
5. **DCN scorer (L6).** Build it as a *benchmarked* alternative to the linear scorer, or keep the interpretable linear + bandit and cite DCN in the brief? (More work vs. cleaner story.)
6. **DQN sequencing (L9).** Attempt the multi-step engagement bonus (star→issue→PR), or keep RL focused on the bandit? My lean: bandit first; DQN only if everything else is solid.
7. **Live-ness on the hosted app.** Confirm: real Kafka locally for the demo + screenshots; hosted app uses the snapshot + a lightweight in-process "live refresh."

---

## 11. Submission deliverables checklist

- [ ] `code/` — all source, `requirements.txt`, single-command run
- [ ] `data/` — dataset (SQLite/CSV) + offline snapshot (≥5k; we target 10k+)
- [ ] `brief.pdf` — ≤4 pages (architecture, techniques + rationale + benchmarks, persona results, limitations)
- [ ] `prompts.md` — key AI prompts + how we modified outputs
- [ ] `README` — exact setup + run instructions
- [ ] Live hosted URL (Streamlit Cloud)
- [ ] Persona pass/fail table (6 capabilities × personas)
- [ ] 10-min demo rehearsed

---

## 12. Phase confirmation log

| Phase | Status | Notes |
|---|---|---|
| 0 — Foundation | ✅ done | |
| 1 — Ingestion & Streaming | ✅ done | 17,227 records, 4 sources, 15 domains |
| 2 — Embedding & Retrieval | ✅ done | all-MiniLM-L6-v2 + FAISS; P@10 0.89 |
| 3 — Relevance & Eval Harness | ✅ done | LLM-as-judge (cached) + NDCG + persona pass/fail; built alongside scoring |
| 4 — Scoring & Ranking | ✅ done | 4 engagement modes + ROI + stratified recall + cross-encoder Stage-2 + MMR + knapsack; **12/12 personas PASS** |
| 5 — Adaptive Learning | ⬜ | after the UI checkpoint |
| 6 — Batch Analytics | ⬜ | |
| 7 — Dashboard & Brief | ⬜ next | user's chosen checkpoint after Phase 3 |
| 8 — Hosting | ⬜ | |
| 9 — Brief, Pass/Fail & Demo | ⬜ | |

> **Sequencing note (user, 2026-06-02):** after Phase 3 we build the **UI** next (good checkpoint to see scoring/ranking in product form), *then* the adaptive RL layer (whose learnings feed back into the model), then iterate on data + UI.
> **Matching finding (Phase 3):** dense embeddings alone underperform a TF-IDF baseline on this short-jargon corpus (0.63 vs 0.77 NDCG); the **cross-encoder Stage-2 re-rank** is the fix (wins for skill-match personas) and is now the pipeline's relevance signal. Open option: try a stronger embedding model (e.g. mpnet-768d) if the brief wants a cleaner dense-vs-keyword win.
