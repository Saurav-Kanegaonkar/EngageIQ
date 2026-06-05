# EngageIQ — Database structure & data flow

EngageIQ uses **two SQLite databases, kept separate on purpose**:

| Database | Mode | Holds |
|---|---|---|
| `data/engageiq.sqlite` | **read-only** | the 20,995 opportunities + their features (the corpus). Built offline, never written at runtime. |
| `data/engage.sqlite` | **read-write** | all per-user state (created at runtime). |

Keeping them apart means the corpus stays a clean, reproducible snapshot while the user data is free to change.

Everything in `engage.sqlite` is scoped to **(user_id, persona_key)**: one browser = one `user_id` (a localStorage id, no login), and each persona a user picks or builds = one `persona_key`. `opportunities` lives in the *other* file, so `opportunity_id` is a logical reference across databases, not an enforced foreign key.

---

## 1. Schema (entity-relationship)

```mermaid
erDiagram
  users ||--o{ personas : "owns"
  personas ||--o{ interactions : "scopes"
  personas ||--o{ events : "scopes"
  personas ||--|| learner_state : "has"
  opportunities ||--o{ interactions : "referenced by id"
  opportunities ||--o{ events : "referenced by id"

  users {
    TEXT user_id PK "the browser id"
    REAL created_at
    REAL last_seen_at
  }
  personas {
    TEXT user_id PK
    TEXT persona_key PK "sofia | custom_ab12"
    TEXT name
    TEXT goal
    TEXT domains "json"
    TEXT platforms "json"
    REAL time_budget_hours
    TEXT mode "contribute|discuss|monitor|promote"
    TEXT split "json: confirmed weekly split"
  }
  interactions {
    TEXT user_id PK
    TEXT persona_key PK
    TEXT opportunity_id PK
    TEXT status "none|interested|dismissed"
    INTEGER saved "0 / 1"
    TEXT reason "why dismissed"
    INTEGER rank
    TEXT bucket
    TEXT source
    TEXT signals "json snapshot"
    REAL first_at
    REAL last_at
  }
  events {
    INTEGER id PK "autoincrement"
    TEXT user_id FK
    TEXT persona_key FK
    TEXT opportunity_id FK
    TEXT type "engage|dismiss|save|view|open_detail"
    TEXT reason
    INTEGER rank
    TEXT bucket
    TEXT signals "json snapshot"
    REAL ts
  }
  learner_state {
    TEXT user_id PK
    TEXT persona_key PK
    TEXT mode
    INTEGER n "reactions seen"
    TEXT weights "json: the RL weight vector"
    REAL updated_at
  }
  opportunities {
    TEXT opportunity_id PK
    TEXT source "github|reddit|..."
    TEXT title
    TEXT url
    TEXT domains "json"
    INTEGER community_size "github stars"
  }
```

### Table by table
- **users** — one row per browser (the localStorage id): who showed up and when.
- **personas** — each profile a user picked or built, plus their confirmed weekly **split**. Rows are upserted, so editing a persona updates the same row in place (and keeps its learning).
- **interactions** — the heart of it. **One row per (user, persona, card)** = the *current* disposition: `status` (none / interested / dismissed), a `saved` flag, the dismissal `reason`, and a snapshot of the card's rank / bucket / source / signals. Idempotent: re-clicking updates this row, never duplicates it. This is what the feed restores on reload and what the Activity view reads.
- **events** — an **append-only history**: every click (engage / dismiss / save / page view / detail open) with the same signal snapshot. The substrate for the "what makes this user engage" analysis.
- **learner_state** — the **reinforcement-learning weight vector** per (user, persona): `n` reactions seen and `weights` (json). This is what tailors the ranking.
- **opportunities** *(in the read-only corpus)* — the source items the others point at by id.

---

## 2. How data flows

```mermaid
flowchart LR
  B["Browser (single-page app)<br/>localStorage: user_id, persona records"]

  subgraph FAST["FastAPI (one service)"]
    direction TB
    PL["POST /api/plan"]
    RE["POST /api/recommend"]
    FE["POST /api/feedback"]
    AC["POST /api/activity"]
    EV["POST /api/event"]
  end

  subgraph CORP["engageiq.sqlite — READ-ONLY corpus"]
    OPP[("opportunities<br/>+ features")]
  end

  subgraph STATE["engage.sqlite — READ-WRITE user state"]
    direction TB
    US[("users")]
    PE[("personas")]
    IN[("interactions")]
    EVT[("events")]
    LS[("learner_state")]
  end

  B --> PL
  B --> RE
  B --> FE
  B --> AC
  B --> EV

  PL -->|read topics| OPP
  RE -->|rank candidates| OPP
  RE -->|read learned weights| LS
  RE -->|upsert| US
  RE -->|upsert + split| PE
  RE -->|read card states| IN
  FE -->|upsert disposition| IN
  FE -->|append| EVT
  FE -->|train + save| LS
  EV -->|append| EVT
  AC -->|read reactions| IN
  AC -->|read learning| LS
  AC -->|read titles/urls| OPP
```

Read vs write at a glance:
- **`/api/plan`** reads the corpus (active topics). No user-state write.
- **`/api/recommend`** ranks against the corpus, reads the learned weights, **writes** `users` + `personas` (upsert), and reads `interactions` to restore each card's state.
- **`/api/feedback`** **writes** `interactions` (upsert), appends `events`, and trains + saves `learner_state`. Idempotent.
- **`/api/event`** appends `events`.
- **`/api/activity`** reads `interactions` + `learner_state` and joins titles/urls from the corpus.

---

## 3. The lifecycle in one line
Pick or build a persona → **`/api/recommend`** persists the persona and reads its learned weights → you react → **`/api/feedback`** upserts `interactions`, appends `events`, and trains `learner_state` → **`/api/activity`** reads it all back so you can see your reactions and what was learned → the next recommend loads the updated weights and re-ranks. The loop is closed.
