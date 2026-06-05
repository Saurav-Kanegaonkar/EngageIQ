# EngageIQ — System Architecture & Data Flow

> How the live system moves data: the API, the two databases, what is written and
> read at every step, the full persona lifecycle (create / update), and the
> engagement pipelines. Companion to `BUILD_SUMMARY.md` (what was built) and
> `PROJECT_LOG.md` (the journey). **Last updated: 2026-06-03.**

---

## 1. The moving parts

```
  Browser SPA  ───HTTP/JSON───►  FastAPI (code/api.py)  ───►  Engine (rank.Recommender)
 (prototype.html)                       │                        │
  localStorage:                         │                        ├─ FAISS index (in memory)
   - eiq-session  (user id)             │                        ├─ embeddings.npy
   - eiq-resume   (last hub)            │                        └─ features (signals)
   - eiq-theme                          │
                                        ├─ reads  ►  engageiq.sqlite   (READ-ONLY corpus)
                                        └─ writes ►  engage.sqlite     (READ-WRITE user state)
```

- **Browser SPA** (`mockups/prototype.html`): onboarding -> plan -> hub, all client-rendered. Holds no source of truth; everything durable lives server-side. Three localStorage keys (below).
- **FastAPI** (`code/api.py`): the only thing the browser talks to. Loads the engine once at startup; owns both databases.
- **Engine** (`code/engageiq/rank.py` + `score.py`, `features.py`, `retrieve.py`): pure scoring/ranking over the corpus. Stateless per request (the only per-user input is the learned weight vector + the confirmed split, both passed in).

---

## 2. Two databases (deliberately separate)

| DB | Mode | Holds | Why separate |
|---|---|---|---|
| `data/engageiq.sqlite` | **read-only** | the 20,995 opportunities + their features | It is the offline snapshot graders run against. Never mutated at runtime. |
| `data/engage.sqlite` | **read-write** | all user state (below) | Mutable per-user data is isolated, so the corpus stays a clean, reproducible artifact. Safe to delete/reset anytime. |

`engage.sqlite` (`code/engageiq/store.py`) — five tables:

```
users         user_id PK, created_at, last_seen_at
personas      (user_id, persona_key) PK, name, goal, domains, platforms,
              time_budget_hours, avoid, mode, split, created_at, updated_at
interactions  (user_id, persona_key, opportunity_id) PK   -- ONE row/card = current state
              status('none'|'interested'|'dismissed'), saved(0/1), reason,
              rank, page, bucket, source, mode, signals(json), first_at, last_at
events        id PK, user_id, persona_key, opportunity_id, type, reason,
              rank, page, bucket, source, mode, signals(json), ts   -- append-only history
learner_state (user_id, persona_key) PK, mode, n, weights(json), updated_at
```

`interactions` = the idempotent **current** disposition (one row per card). `events` = the **full** click history with a signal snapshot per action (the substrate for "why does this user engage"). `learner_state` = the RL weights.

---

## 3. Client-side state (localStorage)

| Key | Set when | Used for |
|---|---|---|
| `eiq-session` | first visit (random id) | the durable **user_id**. Same browser = same user (no login). |
| `eiq-custom-pid` | first custom build | the stable custom persona id (`custom_<random8>`), so edits update in place. |
| `eiq-custom-profile` | on submit | the custom form values, to prefill the form when editing. |
| `eiq-personas` | on entering plan/hub | `{ <persona_key>: {payload, name, ava, split} }` so a refresh / direct link to `/hub/<key>` can reconstruct the screen (the DB then restores states + learning). |
| `eiq-theme` | toggle | light / dark. |

**The URL is the source of truth for navigation** (path-based routing). The app is served as a single page at `/`, `/plan/<persona>`, and `/hub/<persona>`; client-side History API keeps the address in sync (pushState on navigate, popstate for back/forward), and a refresh/direct link re-enters that screen. The app file is `mockups/index.html`; `/prototype.html` 307-redirects to `/`.

---

## 4. API surface (input → reads → writes → output)

| Endpoint | Input | Reads | Writes | Returns |
|---|---|---|---|---|
| `GET /api/personas` | — | `personas.PERSONAS` (static) | — | the 4 named personas for the onboarding cards |
| `POST /api/plan` | `{persona_id \| profile}` | corpus (hot topics), `plan_reads.json` | `plan_reads.json` (LLM read cache only) | the read, recommended split (buckets + %), topics. **No user-state write.** |
| `POST /api/recommend` | `{persona_id\|profile, session_id, k, split}` | `learner_state` (weights), `interactions` (states), corpus + FAISS + features | **`users`, `personas` (upsert)**, `_CTX` (memory cache of signals/bucket/source) | ranked feed + buckets + per-item `state` + `persona_key` + learned summary |
| `POST /api/feedback` | `{session_id, persona_key, opportunity_id, action, rank, page, reason}` | `interactions` (current), `learner_state` | **`interactions` (upsert), `events` (append), `learner_state` (if new disposition)** | new `status`/`saved` + learned summary |
| `POST /api/event` | `{session_id, persona_key, type, opportunity_id, rank, page}` | `_CTX` | **`events` (append)** | ok |
| `POST /api/activity` | `{session_id, persona_key}` | `interactions`, `learner_state`, corpus (titles/urls) | — | the user's reactions grouped `{interested[], saved[], dismissed[], counts, learned}` (powers the Activity view) |
| `POST /api/feedback/clear-all` | — | — | **wipes all 5 tables** | ok |
| `GET /`, `GET /plan/{p}`, `GET /hub/{p}`, `GET /activity/{p}` | — | serves `index.html` | — | the single-page app (so refresh / direct links to a screen work) |
| `GET /prototype.html` | — | — | — | 307 redirect to `/` (legacy) |

Key point: **the only place a persona is persisted today is `/api/recommend`** (via `save_persona`, an upsert keyed by `user_id + persona_key`). `/api/plan` is a pure read/compute.

---

## 5. The data lifecycle, step by step

```
 pick/build persona
        │
        ▼
 POST /api/plan ───────────► read + recommended split + topics      (NO user write)
        │  user tunes the split (sliders, 0% drops a bucket), confirms
        ▼
 POST /api/recommend ──────► engine funnel (pipeline 1)
        │   reads  learner_state (this persona's learned weights)
        │   writes users + personas (upsert: profile + confirmed split)
        │   reads  interactions  (to stamp each card's saved state)
        │   returns ranked feed + buckets + persona_key + per-item state
        ▼
 Hub renders cards ; client saves eiq-resume
        │  user reacts: engage / dismiss(+reason) / save / unsave / undo
        ▼
 POST /api/feedback ───────► record() (pipeline 2)
        │   upsert interactions (idempotent: state flips only on first transition)
        │   append events (every click, with signal snapshot)
        │   if first transition: learner.update(signals, reward) -> save learner_state
        ▼
 next /api/recommend loads the updated weights -> re-ranks   (the learning loop closes)

 refresh at any time -> eiq-resume reopens the Hub -> /api/recommend restores states + weights
```

### 5.1 New persona created
- **Named** (pick Sofia): `persona_key = "sofia"` (stable). Persisted on the first `/api/recommend`.
- **Custom** (build-your-own form): `persona_key = "custom:" + sha1(goal + interests)`. Persisted on the first `/api/recommend`.
- Learning + history are scoped to `(user_id, persona_key)` from that point on.

### 5.2 Persona updated  (✅ stable id, see §8)
- `save_persona` is an **UPSERT** on `(user_id, persona_key)`. Custom personas now carry a **stable id** (minted once, `localStorage["eiq-custom-pid"]`), so editing the goal/domains/split **updates the same row in place** and keeps the learning + interactions + events.
- If the edit changes the **engagement mode**, the learner is reset to the new mode's defaults (history kept) and the Hub shows a "focus changed" notice. Named personas keep their fixed ids; "Edit & customize" on a named persona forks it into the user's custom persona.

### 5.3–5.6
- **Plan view:** pure compute; the LLM "read" is cached to `plan_reads.json`.
- **Hub / recommend:** the funnel (pipeline 1) + persona upsert + state restore.
- **React to a card:** the feedback loop (pipeline 2), idempotent.
- **Refresh:** `eiq-resume` reopens the Hub; the DB restores card states + RL weights.

---

## 6. The pipelines

### 6.1 Recommendation funnel (`rank.Recommender.recommend`)
```
profile ─► multi-facet query (goal + each interest)
        ─► FAISS stratified per-source recall (top 150 / allowed source)      [allowed = active buckets' sources]
        ─► hard filters (platform, domain whitelist, excluded languages)
        ─► cross-encoder relevance re-rank (Stage-2, top ~300)
        ─► score: impact = Σ weight·signal   [weights = learned-per-user OR mode defaults]
                  · contribute mode: ×worthiness gate (down-ranks random/abandoned repos)
                  · ×per-bucket prior (from the confirmed split)
        ─► sort by impact
        ─► MMR diversity re-rank -> top k
        ─► attach "why this" + action
        ─► [time-budget knapsack basket | velocity radar] + standout-repo lane
returns: ranked, buckets, rail, repos, persona_key, per-item state, learned summary
```
The client then **interleaves** `ranked` by the confirmed split (weighted round-robin) for the "For you" feed, and each bucket chip filters to its own items.

### 6.2 Feedback → learning loop (`store.record` + `feedback.Learner`)
```
action ─► look up the card's signal vector (server cache _CTX, not the client)
       ─► upsert interactions (status/saved/reason)      [idempotent: no double-count]
       ─► append events (with the signal snapshot)
       ─► if first transition for this disposition:
            reward = engage 1.0 | save 0.8 | dismiss 0.0
            w += lr · (reward − w·signals) · signals ; clip ≥0 ; renormalize to sum 1   (online LMS / Widrow-Hoff)
            save learner_state
```
Interpretable by construction: the weights stay non-negative and sum to 1, so "Why this?" reads straight off them, and the learning is visible in the toast.

### 6.3 Analytics / "why" (not built yet — what the `events` table is for)
Aggregate `events` per `(user, persona)`: which signals / buckets / ranks the user engages with vs dismisses -> a "what makes you engage" view + week-on-week growth + the Trends tab (corpus-level trends via PySpark + Count-Min / HyperLogLog).

---

## 7. Identity & scoping
- **user_id** = the browser's `eiq-session` (no accounts; one browser = one user). Pragmatic for a demo, and the only privacy-safe option here.
- Everything user-facing is scoped to **(user_id, persona_key)**, because the RL weights are engagement-mode specific. Two people, or one person's two personas, never mix.
- The schema already supports **many personas per user** (PK is composite). The UI does not yet list "your personas", but the data model is ready for it.

---

## 8. Persona identity: stable id, update in place  ✅ RESOLVED (2026-06-03)

**Was:** a custom persona's `persona_key` was a hash of its goal + interests, so editing it forked a new persona and orphaned the old one's learning.

**Now:** each custom persona has a **stable id minted once at creation** (`customPid()` -> `localStorage["eiq-custom-pid"]` -> `"custom_<random8>"`), sent as `profile.id` on every call; `_persona_key()` returns it directly (the content hash is only a legacy fallback). So:
1. Editing a persona **upserts the same row** (goal/domains/split change in place); its learning + interactions + events stay attached.
2. If an edit changes the **engagement mode** (e.g. contribute -> monitor), `save_persona` **resets the learner** to the new mode's defaults (old weights were for a different signal mix) but **keeps the interaction/event history**, and `/api/recommend` returns a `notice` the Hub shows ("your focus changed, so learning restarts; your saved cards + history are kept").
3. The form **prefills** from `localStorage["eiq-custom-profile"]` when you edit, so you see your current values.

Verified end-to-end (curl): same-mode edit keeps `n`; mode-change edit updates the one row in place, resets the learner, and preserves interactions.

**Still deferred (user choice):** a "your personas" switcher + persisting at create/plan time (the schema already supports many personas per user).

---

## 9. What's solid vs what to revisit
- **Solid:** two-DB split, idempotent interactions, append-only events with signal snapshots, RL persisted + reloaded per (user, persona), client resume, the recommend funnel, **stable persona identity / update-in-place (§8)**.
- **Revisit (deferred):** persist at create/plan time vs first-recommend; a "your personas" switcher (schema-ready); then the analytics pipeline (§6.3).

---

## 10. Hosting (decisions for the live URL)
- **One service, one origin.** FastAPI serves both the SPA and the JSON API, so there is no CORS setup and a single process is the whole product.
- **Run with a single worker.** The per-session signal cache (`_CTX`) that backs `/api/feedback` is in-process, so feedback must hit the same worker that served the feed. `uvicorn api:app --app-dir code` defaults to 1 worker, which is correct and plenty for demo traffic. (To go multi-worker later, move `_CTX` into the DB.)
- **Keyless / offline for graders.** The corpus (`engageiq.sqlite`) and the LLM card summaries (`data/summaries.json`) are bundled; the LLM is optional everywhere (deterministic fallbacks for the plan read, cached summaries for cards). The professor needs no API keys; the `NVIDIA_API_KEY` only adds live LLM reads for brand-new custom personas.
- **User DB is ephemeral, and that is fine.** `engage.sqlite` is created at runtime; on a free host it may reset on redeploy/sleep. Reactions + learning persist within a session; the read-only corpus is always bundled, so the core demo always works.
- **Served folder = `mockups/` holds only `index.html`.** The old design mockups + the card-comparison page were moved to `archive/` (not served), so the live URL exposes only the product and `/api/*`.
