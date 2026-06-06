# World Cup 2026 — Agent System Deep-Dive Analysis

> **Date**: 2026-06-06 | **Author**: AI System Analysis  
> **Goal**: Understand the current agent architecture end-to-end, audit capabilities vs. the target user experience, and produce a scored, prioritized improvement roadmap.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Agent-by-Agent Deep Dive](#2-agent-by-agent-deep-dive)
   - [Orchestrator (`orchestrator.py`)](#21-orchestrator)
   - [Planner Agent (`planner_agent.py`)](#22-planner-agent)
   - [BigQuery Agent (`bigquery_agent.py`)](#23-bigquery-agent)
   - [Match Facts Agent (`match_facts_agent.py`)](#24-match-facts-agent)
   - [Prediction Agent (`prediction_agent.py`)](#25-prediction-agent)
   - [News Agent (`news_agent.py`)](#26-news-agent)
   - [Sentiment Agent (`sentiment_agent.py`)](#27-sentiment-agent)
   - [Result Composer (`result_composer_agent.py`)](#28-result-composer)
   - [Docs Agent, Code Review Agent, Workflow Logger](#29-supporting-agents)
3. [Data Model Deep Dive (BigQuery)](#3-data-model-deep-dive)
4. [Communication Flow](#4-communication-flow)
5. [Question Capability Matrix](#5-question-capability-matrix)
6. [API Call Strategy & Cost Optimization](#6-api-call-strategy--cost-optimization)
7. [Improvement Roadmap](#7-improvement-roadmap)
8. [Progress Scorecard](#8-progress-scorecard)

---

## 1. Architecture Overview

```
User Message (WhatsApp / Streamlit / Web)
        │
        ▼
┌─────────────────────────────────────┐
│         ORCHESTRATOR                │
│  (LangGraph StateGraph pipeline)     │
│                                     │
│  classify_intent ──► route_request  │
│       │                   │         │
│       ▼                   ▼         │
│  execute_agents ──► aggregate       │
│       │                   │         │
│       ▼                   ▼         │
│  score_confidence ──► compose_reply │
└─────────────────────────────────────┘
        │
        ▼
┌───────────────────┐    ┌──────────────────┐
│   PLANNER AGENT   │───►│  SPECIALIST       │
│  (agent selection) │    │  AGENTS           │
│                   │    │                   │
│  Rules:           │    │  • bigquery       │
│  - structured     │    │  • match_facts    │
│    data → bigquery│    │  • prediction     │
│  - prediction →   │    │  • news           │
│    pred + bq      │    │  • sentiment      │
│  - news → news    │    │  • chat (fallback)│
│  - sentiment →    │    │                   │
│    sentiment      │    └──────────────────┘
│  - chat → chat    │
└───────────────────┘
```

**Key observations**:

- **LangGraph pipeline**: 6 sequential nodes (`classify` → `router` → `execute_agents` → `aggregate` → `confidence` → `compose`). No branching, no conditional edges. This is a straight pipeline — not a graph.
- **Planner selects 1-2 agents**. The orchestrator runs them **sequentially** (line 236-250 in orchestrator), not in parallel. This is correct for cost control but increases latency for multi-agent queries.
- **No streaming**. The entire pipeline must complete before the user sees anything. For a fan watching a live game, waiting 15-30 seconds is unacceptable.
- **match_facts_agent is NOT in the planner's AVAILABLE_AGENTS list** but is registered in the orchestrator's `runners` dict and has an `_run_match_facts` function. It's **unreachable** via normal routing — a dead code path.

---

## 2. Agent-by-Agent Deep Dive

### 2.1 Orchestrator

**File**: `src/agents/orchestrator.py`  
**Model**: `gpt-4o-mini`  
**State**: `OrchestratorState` TypedDict with 12 fields

**What it does well**:
- Clean separation of concerns: classification, routing, execution, aggregation, confidence, composition are distinct nodes.
- The `aggregate_outputs_node` has a smart priority system: BigQuery-backed outputs always win over API/web outputs when synthesizing multi-agent results. This is **correct and critical** for factual accuracy.
- `_pick_primary_payload` ranks agents by data source quality (bigquery=3, api=1, other=0) then by confidence score. Good design.
- WorkflowTracker provides full execution tracing.

**Issues & Risks**:

- **[Severity: High]** `match_facts_agent` is defined as a runner (`_run_match_facts`) but is **NOT in the planner's `AVAILABLE_AGENTS`** list. The planner can never select it. All match-facts queries route to `bigquery` instead. The orchestrator's `_run_match_facts` function is dead code.
- **[Severity: Med]** The graph is a straight pipeline — no conditional branching. The `intent` field from `classify_intent` is stored in state but **never used for routing decisions**. The `route_request` node calls the planner regardless of the intent classification. Two LLM calls for classification is wasteful.
- **[Severity: Med]** `_AGENTS` list includes `"bigquery"` but the intent classifier uses `_INTENTS = ["news", "sentiment", "data", "prediction", "chat"]`. The `"data"` intent doesn't map directly to `"bigquery"` agent name — the planner bridges this, but it's a fragile disconnect.
- **[Severity: Low]** The orchestrator uses `gpt-4o-mini` for classification, planning, AND chat responses. For production, consider a faster/cheaper model for classification (classification is a simple 5-way choice).

### 2.2 Planner Agent

**File**: `src/agents/planner_agent.py`  
**Model**: `gpt-4o-mini`  
**Purpose**: Selects 1-2 specialist agents for a given user query.

**What it does well**:
- Has a robust `_fallback_plan` with keyword-based routing when JSON parsing fails. This is excellent defensive design.
- Returns structured JSON with `agents`, `response_mode`, `reason`, `primary_agent`.
- Enforces the critical rule: "bigquery is the single source of truth for all structured football data."

**Issues & Risks**:

- **[Severity: High]** `AVAILABLE_AGENTS = ["news", "sentiment", "prediction", "bigquery", "chat"]` — **`match_facts` is missing**. This means queries like "What's the lineup for Portugal vs Morocco?" or "Who is the referee?" route to `bigquery` instead of `match_facts`. The `bigquery_agent` can answer these from the data model, but `match_facts_agent` has API-Football integration with live data that the gold model may not have yet.
- **[Severity: Med]** The planner prompt says "Never select more than 2 agents unless truly necessary" but also says to include `prediction + bigquery` for prediction queries. For a question like "predict Portugal vs Morocco and give me recent news about both teams", it would need 3 agents (prediction, bigquery, news) but the planner would truncate to 2. The orchestrator's `selected_agents = cleaned[:3]` does allow up to 3, creating an inconsistency.
- **[Severity: Low]** The planner has no concept of "the user just asked a follow-up." It treats every message independently. Context awareness is limited to the conversation history text passed in the prompt — no structured turn tracking.

### 2.3 BigQuery Agent

**File**: `src/agents/bigquery_agent.py`  
**Model**: `gpt-4o-mini` with function calling  
**Max tool turns**: 8

This is the **most important agent** in the system. It's the single source of truth for all structured football data.

#### How It Works (Function-Calling Loop)

```
User Query: "What is Portugal's recent form?"
    │
    ▼
┌──────────────────────────────────────┐
│  System Prompt with FULL catalog     │
│  (all marts, facts, dims inline)     │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────┐    ┌──────────────┐
│ 1. list_tables() │───►│ See all      │
│    (optional)    │    │ available    │
└──────────────────┘    │ tables       │
                        └──────────────┘
    │
    ▼
┌──────────────────┐    ┌──────────────┐
│ 2. describe_table│───►│ Get schema   │
│   ("mart_team_   │    │ + usage hint │
│    form")        │    │ for the table│
└──────────────────┘    └──────────────┘
    │
    ▼
┌──────────────────┐    ┌──────────────┐
│ 3. sample_table()│───►│ See 5 rows   │
│    (optional)    │    │ of real data │
└──────────────────┘    └──────────────┘
    │
    ▼
┌──────────────────┐    ┌──────────────┐
│ 4. run_sql()     │───►│ Execute      │
│    SELECT ...     │    │ validated    │
│    FROM mart_team │    │ read-only    │
│    _form ...      │    │ query        │
└──────────────────┘    └──────────────┘
    │
    ▼
┌──────────────────┐
│ 5. Format answer │
│    (grounded in  │
│     retrieved    │
│     rows)        │
└──────────────────┘
```

#### Data Model (Catalog)

The BQ agent operates on a **star schema** with 3 layers:

**Marts (preferred — agent should start here)**:

| Table | Grain | Key Use Case |
|-------|-------|-------------|
| `mart_team_profile` | 1 row per team | All-time team stats (win%, goals, clean sheets) |
| `mart_team_form` | 1 row per team | Last 10 match form (WDL string, points, GF/GA) |
| `mart_head_to_head` | 1 row per unordered pair | Head-to-head aggregates between two teams |
| `mart_match_history` | 1 row per completed match | Past matches with wide-pivoted home/away stats |
| `mart_match_upcoming` | 1 row per upcoming/live match | Next fixtures with pre-enriched form + H2H |
| `mart_tournament_state` | 1 row per team/comp/season | Current standings + next/last match |

**Facts (fallback when no mart fits)**:

| Table | Grain | Key Use Case |
|-------|-------|-------------|
| `fact_match` | 1 row per match_id | Canonical match record (score, status, venue) |
| `fact_match_team` | 1 row per (match, team) | Per-team stats (possession%, shots, xG, W/D/L) |
| `fact_match_event` | 1 row per (match, event_seq) | Goals, cards, substitutions, VAR events |
| `fact_standings_snapshot` | 1 row per (team, date, comp) | Historical standings snapshots |

**Dimensions (for name → ID resolution)**:

| Table | Purpose |
|-------|---------|
| `dim_team` | Team master with `is_wc2026_participant` flag |
| `dim_competition` | Competition master (WC = competition_id=1) |
| `dim_venue` | Venue master with city, capacity |
| `dim_date` | Calendar dimension |

There's also a **source-level** table `fact_fixture` used by `api_football.py` and `predict.py` — it's a raw API mirror, not part of the catalog, but queried directly by the match_facts and prediction agents (bypassing the catalog system).

#### Types of Queries the BQ Agent Can Generate

**✅ Can answer accurately**:
- Team form (last N matches from `mart_team_form`)
- Head-to-head records (`mart_head_to_head`)
- All-time team stats (`mart_team_profile`)
- Upcoming fixtures (`mart_match_upcoming`)
- Past match results with stats (`mart_match_history`)
- Tournament standings (`mart_tournament_state`)
- Match events: goals, cards, substitutions (`fact_match_event`)
- Per-team match stats: possession, shots, xG (`fact_match_team`)
- Any ad-hoc aggregation over these tables

**⚠️ Can answer with limitations**:
- Lineup queries: `fact_match_event` has player names in events but no structured lineup (11 players with positions). The agent would need to reconstruct a lineup from event data, which is unreliable.
- Live/in-play data: All tables require data to be ingested first. If the ETL hasn't run, live data is stale.
- Player-specific stats: There's no `dim_player` or player-level fact table. The agent can only answer about players through `fact_match_event` (event-level) which is coarse.
- Referee queries: `dim_referee` is mentioned in `DATA_CONTRACT.md` but isn't in the catalog. The `fact_fixture` source table has a `referee` column but isn't catalog-visible.

**❌ Cannot answer**:
- "Who is the player with the most shots on target in this game?" — No per-player stat fact table
- "What substitutions have happened in this match and what is the coach trying to do?" — Substitutions are in `fact_match_event` but tactical analysis requires LLM reasoning (not BQ)
- "Who is the top player to watch?" — No player performance metrics
- Weather at venue — Weather is handled by `match_facts_agent` + OpenWeatherMap, not BQ

#### Guardrails (in `datamodel_tools.py`)

- SELECT/WITH only — no DDL/DML
- Single statement — no semicolons
- Table allow-listing — every backticked reference must match an `agent_visible` table
- Auto-LIMIT when missing (default 500 rows)
- 1 GB max bytes billed safety cap
- Forbidden keywords: INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE, MERGE, GRANT, REVOKE, CALL

These guardrails are **excellent**. The allow-listing via the catalog is a particularly strong design choice.

#### Confidence Scoring

- No SQL calls → 0.3 ("Agent answered without querying BigQuery")
- All SQL calls failed → 0.3
- Queries ran but 0 rows → 0.5
- Queries returned rows → 0.85 (regardless of whether the answer is actually correct!)
- The confidence score does **NOT** validate semantic correctness — it only checks if SQL ran and returned rows. A query could return wrong data and still get 0.85.

### 2.4 Match Facts Agent

**File**: `src/agents/match_facts_agent.py`  
**Model**: `gpt-4o-mini`

**What it does**: Fetches fixtures (via `get_fixtures_cache_first`), enriches with weather, formats with LLM. Has extensive web-search fallback.

**What it does well**:
- Cache-first architecture: BigQuery → API-Football → web search → LLM fallback
- Handles countdown queries ("How many days until World Cup?") with hardcoded calendar logic (score: 0.98)
- Deduplicates fixtures by (id, date, home, away)
- Has both LLM-based formatting and a template fallback
- `_summarise_web_hits` is sophisticated: extracts sentences from scraped pages, scores by keyword match, falls back to LLM summarization

**Issues & Risks**:

- **[Severity: High]** This agent should be the **primary agent for all match-specific queries** (lineups, venue, referee, weather) but it's **not in the planner's agent list**. All those queries go to `bigquery_agent` instead.
- **[Severity: Med]** The agent fetches web pages (`_fetch_page_snippet`) which takes 1-3 seconds per URL. For "today's matches," this could mean 3+ HTTP requests adding 5-9 seconds of latency.
- **[Severity: Med]** Weather is only fetched for the first fixture when `wants_list=False`. For "today's matches" (list mode), no weather is fetched.
- **[Severity: Low]** The `_normalise_fixture_with_season` function exists in both `match_facts_agent.py` (not used there) and `api_football.py`. Duplication.

### 2.5 Prediction Agent

**File**: `src/agents/prediction_agent.py`  
**Model**: `gpt-4o-mini` (for reasoning text)

**How it works**:

```
User: "Predict Portugal vs Morocco"
    │
    ▼
┌──────────────────────────────────────┐
│  predict_match() in predict.py       │
│                                      │
│  1. Parse matchup from query         │
│  2. Try BQ heuristic (cache-first)   │
│     - Fetch last 10 for each team    │
│     - Fetch H2H history              │
│     - Compute: PPM, goal diff,       │
│       H2H edge → sigmoid calibration │
│  3. If BQ miss: warm cache from API  │
│     → retry BQ heuristic             │
│  4. If still no data: uniform 33/33/33│
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  LLM generates 2-sentence tactical   │
│  reasoning based on probabilities    │
└──────────────────────────────────────┘
```

**What it does well**:
- The heuristic is statistically sound: uses a sigmoid calibration with points-per-match, goal difference, and H2H edge as features
- Cache-first design minimizes API costs
- Graduated confidence: BQ cache=0.85, API-then-BQ=0.7, uniform fallback=0.4
- `_warm_cache_from_api` only fetches 2 seasons (2018, 2022), not all history

**Issues & Risks**:

- **[Severity: High]** The prediction model (`_MODEL_PATH`) is expected at `bin/models_deployed/wc2026_predictor.pkl`. If this file doesn't exist (which it probably doesn't yet), it falls through to `_heuristic_probs`. The XGBoost path (`model_version: xgboost_features_missing_fallback`) returns uniform 33/33/33 — it's a TODO, not a real model.
- **[Severity: Med]** The heuristic queries `fact_fixture` (source table) directly, **bypassing the catalog system**. This means the prediction agent has its own SQL, its own table references, and its own query logic — completely separate from the BQ agent's catalog-driven approach. Two different code paths for querying the same data.
- **[Severity: Med]** No odds integration. The user asked for "fetch the odds for the respective game" — this requires an odds API (e.g., The Odds API) which isn't integrated.
- **[Severity: Low]** The `predict_match` function in `predict.py` also bypasses team name resolution through `dim_team`. It uses `LOWER(home_team_name) LIKE '%team%'` pattern matching which is fragile for teams with similar names (e.g., "Korea Republic" vs "Korea DPR").

### 2.6 News Agent

**File**: `src/agents/news_agent.py`  
**Model**: None (uses search tools only)

**What it does**: Searches for news via the `search_news` tool with a fallback chain: DuckDuckGo HTML → Serper → Tavily → NewsAPI → Google News RSS.

**What it does well**:
- 5-provider fallback chain with free options (DuckDuckGo, Google News RSS) always available
- Confidence scales with article count: `min(0.95, 0.55 + 0.08 * len(articles))`

**Issues & Risks**:

- **[Severity: Med]** The agent returns article titles + URLs + sources. There's **no content extraction or summarization**. The user gets link lists, not synthesized news. For a live-game companion, users won't click links.
- **[Severity: Low]** No temporal filtering. "Latest news about Portugal" could return articles from 2018. No date-based filtering in the search.

### 2.7 Sentiment Agent

**File**: `src/agents/sentiment_agent.py`  
**Model**: None (uses VADER + Twitter API)

**What it does**: Fetches tweets about a topic, runs VADER sentiment scoring, returns positive/negative/neutral breakdown.

**Issues & Risks**:

- **[Severity: High]** This agent is **unlikely to work in production**. Twitter/X API v2 has severely restricted free-tier access. The `twitter_sentiment.py` tool is not in the workspace (only the import exists). This agent will likely fail or return 0 tweets, yielding confidence 0.2.
- **[Severity: Med]** Sentiment analysis on tweets during a live match has very low signal-to-noise. Fans tweet emotionally — "Portugal is killing it!" could be sarcastic. VADER doesn't handle sarcasm. This agent's output is entertainment, not insight.

### 2.8 Result Composer

**File**: `src/agents/result_composer_agent.py`  
**Model**: `gpt-4o-mini`

**What it does**: Takes the raw agent output and formats it for display. Adds confidence line, low-confidence tips, and prediction caution.

**What it does well**:
- BigQuery answers pass through directly (no re-formatting) to preserve data fidelity
- Non-BQ answers get a formatting pass for chat/mobile readability
- Confidence labels (HIGH/MEDIUM/LOW) with percentages are clear and actionable
- Prediction-specific caution for scores < 0.55

**Issues & Risks**:

- **[Severity: Low]** The formatting prompt says "Start with a direct answer in one short sentence" but BigQuery answers skip this (passed through raw). This means BQ answers might not have the punchy one-liner that mobile users expect.
- **[Severity: Low]** No source attribution. The user can't tell which data came from BQ vs web search vs API. "Source: BigQuery" would build trust.

### 2.9 Supporting Agents

| Agent | Purpose | Status |
|-------|---------|--------|
| `docs_agent.py` | Logs every session to `bin/docs/` as markdown | ✅ Working, called from orchestrator |
| `code_review_agent.py` | Lints + type-checks generated code via Ruff + mypy | ⚠️ Only used when agents generate Python (rare) |
| `workflow_logger.py` | Tracks orchestrator node execution with timestamps | ✅ Working, used extensively |

---

## 3. Data Model Deep Dive

### 3.1 Current State

The data model has **two tiers**:

**Tier 1 — Gold Semantic Model** (catalog-driven, agent-visible):
- 4 dims, 4 facts, 6 marts
- Catalog-driven discovery
- Properly documented with grain, usage hints, example questions
- Enforced by `datamodel_tools.py` guardrails

**Tier 2 — Source Tables** (catalog-invisible, queried directly):
- `fact_fixture` — used by `match_facts_agent` and `prediction_agent`
- Mentioned in `DATA_CONTRACT.md` as a "canonical object" but **not** in the catalog

### 3.2 Critical Gaps

| Gap | Impact | Required For |
|-----|--------|-------------|
| **No `dim_player`** | Cannot answer any player-specific question | "Who is the top scorer?" "Who has most shots on target?" |
| **No `fact_player_match_stat`** | No per-player stats per match | "Player X stats this game" |
| **No `dim_referee`** (in catalog) | Referee data only in source table | "Who is the referee for Portugal vs Morocco?" |
| **No `mart_player_form`** | No recent player performance aggregation | "Player to watch?" "Player X last 5 games" |
| **No `fact_lineup`** | No structured starting XI | "What's the lineup?" |
| **No live match state** | No in-play data (current minute, live score updates) | "What's happening now in the game?" |
| **No odds table** | No betting odds integration | "What are the odds?" |
| **No `dim_coach`** | No coach/manager information | "Who is the coach?" "What's the coach's style?" |

### 3.3 DATA_CONTRACT.md vs. Reality

The contract defines 8 canonical tables but only 4 are in the catalog (`dim_team`, `dim_competition`, `dim_venue`, `fact_fixture` — the last as a source table). The contract also mentions `dim_referee` which doesn't exist in the codebase. Gold views (`v_team_recent_form`, `v_head_to_head`, etc.) are referenced in the contract but the actual implementation uses marts (`mart_team_form`, `mart_head_to_head`, etc.). **The contract is out of date**.

---

## 4. Communication Flow

### Agent-to-Agent Communication

All agents communicate through a **shared state dictionary** (`OrchestratorState`), not through direct agent-to-agent messages. The flow is:

```
User Message
    │
    ▼
classify_intent()          → writes state["intent"]
    │
    ▼
route_request()            → calls planner → writes state["selected_agents"]
    │
    ▼
execute_agents_node()      → loops over selected_agents, calls each
    │                         specialist.run_structured()
    │                         writes state["agent_outputs"][agent_name]
    ▼
aggregate_outputs_node()   → picks primary, optionally synthesizes
    │                         writes state["agent_payload"]
    ▼
score_confidence()         → normalizes score, writes state["confidence_*"]
    │
    ▼
compose_reply()            → result_composer_agent.compose()
                              writes state["final_reply"]
```

**Contract per agent**: Every specialist agent must implement `run_structured(query: str) -> dict` with keys `answer`, `confidence_score`, `confidence_reason`, `metadata`. This is a good, clean contract.

**What's missing**:
- Agents cannot call other agents. If `bigquery_agent` realizes it needs web search, it cannot delegate to `news_agent`.
- No shared memory/cache between agent calls. If `bigquery_agent` resolves "Portugal" → `team_id=5`, the `prediction_agent` re-resolves it independently.
- No streaming between agents. The `aggregate_outputs_node` waits for ALL agents to complete before composing.

---

## 5. Question Capability Matrix

### User's Target Questions — Can the App Answer Them Today?

| # | Question | Can Answer? | Accuracy | Which Agent | Notes |
|---|----------|------------|----------|-------------|-------|
| 1 | "What matches are going to happen today?" | ⚠️ Partial | Medium | BQ via `mart_match_upcoming` | Only works if ETL has ingested today's fixtures. `match_facts_agent` could do live API fallback but isn't routable. |
| 2 | "Where is the stadium?" | ⚠️ Partial | Medium | BQ via `dim_venue` | Venue exists in catalog but join path from match → venue isn't direct in marts. |
| 3 | "How is the weather?" | ⚠️ Partial | Low-Medium | BQ (no weather) / match_facts (has weather) | Weather only works if `match_facts_agent` is routed (it isn't). Even then, OpenWeatherMap is behind `ENABLE_WEATHER=false` by default. |
| 4 | "Who is the referee?" | ⚠️ Partial | Low | BQ via `fact_fixture` (source) | Referee in source table, not catalog marts. Agent might not find it. |
| 5 | "What's Team A's current form?" | ✅ Yes | High | BQ via `mart_team_form` | This is the best-supported query type. |
| 6 | "What's the lineup?" | ❌ No | — | — | No `fact_lineup` or structured lineup data exists. |
| 7 | "Who is the player with the most goals in last 10 games?" | ❌ No | — | — | No `dim_player`, no `fact_player_match_stat`. |
| 8 | "Who has the most shots on target in this game?" | ❌ No | — | — | No per-player in-game stats. |
| 9 | "What substitutions happened and what's the coach doing?" | ❌ No | — | — | Sub events exist in `fact_match_event` but no tactical reasoning capability + no coach data. |
| 10 | "Who's the player to watch? Key stats?" | ❌ No | — | — | No player performance metrics, no "player to watch" logic. |
| 11 | "Pre-match summary with stats and players to watch" | ❌ No | — | — | Requires player data + player form marts. |
| 12 | "Run prediction model and fetch odds" | ⚠️ Partial | Low | Prediction agent | Heuristic works, but no real ML model deployed, no odds API. |

### Other Questions the App CAN Answer Today

| Question | Agent | Accuracy |
|----------|-------|----------|
| "What's Portugal's all-time win rate?" | BQ via `mart_team_profile` | High |
| "How many times have Argentina and Brazil played?" | BQ via `mart_head_to_head` | High |
| "Show Argentina's last 5 matches" | BQ via `mart_match_history` | High |
| "What are the current WC2026 group standings?" | BQ via `mart_tournament_state` | High |
| "When does the World Cup start?" | match_facts (calendar) | Very High (0.98) |
| "What's the latest news about Portugal?" | news_agent | Medium |
| "What's the social sentiment about Morocco?" | sentiment_agent | Low (API restrictions) |
| "Predict Portugal vs Morocco" | prediction_agent | Medium (heuristic) |

---

## 6. API Call Strategy & Cost Optimization

### Current API Usage

| API | Called By | Frequency | Cost Risk |
|-----|-----------|-----------|-----------|
| API-Football (`/fixtures`) | `match_facts_agent`, `prediction_agent` | On cache miss | Medium — 100 req/day free tier |
| OpenWeatherMap | `match_facts_agent` | Per match (if enabled) | Low — 1000 req/day free tier |
| Tavily Search | `news_search.py` | Per news query | Medium — usage-based |
| Serper (Google) | `news_search.py` | Per news query | Medium — usage-based |
| NewsAPI | `news_search.py` | Per news query | Low — 100 req/day free tier |
| DuckDuckGo HTML | `news_search.py` | Per news query | Free |
| Google News RSS | `news_search.py` | Per news query | Free |
| Twitter/X API v2 | `twitter_sentiment.py` | Per sentiment query | High — very restricted free tier |

### Recommended API Call Schedule

| Frequency | API Calls | Rationale |
|-----------|-----------|-----------|
| **Once (pre-tournament)** | All historical data (API-Football: fixtures, teams, venues for 2018, 2022 seasons) | Populate the BQ gold model. One-time cost. |
| **Daily** | API-Football: today's fixtures, tomorrow's fixtures | Keep `mart_match_upcoming` fresh. ~2-3 calls/day. |
| **Every 5 min (during matches)** | API-Football: live fixtures (match events), live standings | Only when matches are LIVE. ~12 calls/hour/match. |
| **Weekly** | API-Football: standings snapshot, team stats refresh | Keep historical snapshots. ~5 calls/week. |
| **On-demand (user query)** | Web search (DuckDuckGo first, then Tavily/Serper as fallback) | Only when user explicitly asks for news. Free tier first. |
| **On-demand (user query)** | Weather (OpenWeatherMap) | Only when user asks about a specific venue's weather. |

### Cost Optimization Recommendations

1. **Cache-first always**. The current `get_fixtures_cache_first` pattern is correct. Extend to ALL API calls.
2. **Batch pre-fetch during live matches**. Don't wait for user query → API call. A cron job should pull API-Football every 5 minutes during match windows and push to BQ. User queries then hit BQ only.
3. **Weather: pre-compute, don't query per user**. Fetch weather for all venues with matches today ONCE, store in BQ, serve from cache.
4. **News: free providers first**. DuckDuckGo HTML and Google News RSS are free. Use them for 80% of news queries, fall back to Tavily/Serper only when free results are insufficient.
5. **Twitter sentiment: disable by default**. The ROI is terrible. Replace with Reddit API (free) or in-app reactions/feedback.

---

## 7. Improvement Roadmap

Below is the prioritized list of improvements to take this app from its current state to the "go-to companion for a fan watching a World Cup game."

### Phase 1 — Fix the Foundation (Must-Do, 2-3 weeks)

| # | Improvement | Effort | Impact | Description |
|---|------------|--------|--------|-------------|
| **P1.1** | Add `match_facts` to planner agents list | 5 min | 🔴 Critical | Unblocks all match-specific queries. One-line change in `planner_agent.py:AVAILABLE_AGENTS`. |
| **P1.2** | Create `dim_player` + `fact_player_match_stat` | 3 days | 🔴 Critical | The single biggest gap. Without player data, 40% of target questions are impossible. Model: one row per (match, player, team) with goals, assists, shots, passes, xG, minutes_played. |
| **P1.3** | Create `fact_lineup` | 1 day | 🔴 Critical | Structured starting XI per match. Table: (match_id, team_id, player_id, position, is_starter, jersey_number). |
| **P1.4** | Create `mart_player_form` | 1 day | 🟡 High | Rolling player performance over last 5 games. Aggregates `fact_player_match_stat`. Enables "player to watch" queries. |
| **P1.5** | Add streaming to orchestrator | 3 days | 🟡 High | Use SSE (Server-Sent Events) or WebSocket. Stream each pipeline step: "Classifying intent... → Routing to BigQuery... → Querying data... → Composing answer...". Users see progress within 1 second. |
| **P1.6** | Wire `match_facts_agent` for live API fallback | 1 day | 🟡 High | When BQ has no data for today's match (ETL not yet run), fall back to API-Football live endpoint. Critical for game-day use. |
| **P1.7** | Fix orchestrator to eliminate redundant classification | 2 hours | 🟢 Med | Remove `classify_intent` node or use its output in routing. Currently both `classify_intent` AND planner run — two LLM calls for the same decision. |

### Phase 2 — Enable the Game-Day Experience (Should-Do, 3-4 weeks)

| # | Improvement | Effort | Impact | Description |
|---|------------|--------|--------|-------------|
| **P2.1** | Build pre-match summary agent | 3 days | 🔴 Critical | New agent that composes: team form (BQ), H2H (BQ), key players to watch (new `mart_player_form`), venue info (BQ), weather (pre-cached API). This is THE core user experience for "I'm about to watch a game." |
| **P2.2** | Live match event polling cron | 2 days | 🟡 High | Cloud Scheduler cron that runs every 5 min during match windows. Calls API-Football fixtures+events endpoints, pushes to BQ `fact_match_event`. Users always get fresh data. |
| **P2.3** | Add `dim_coach` + coach tactical profile | 2 days | 🟡 High | Coach master with: name, nationality, preferred_formation, style_tags. Enables "what is the coach trying to do?" analysis. |
| **P2.4** | Substitution analysis agent | 2 days | 🟡 High | When user asks "What does this sub mean?", agent: (1) fetches sub event from `fact_match_event`, (2) fetches player profiles of subbed-in/out players, (3) fetches coach profile, (4) uses LLM to generate tactical reasoning. |
| **P2.5** | Deploy real ML prediction model | 5 days | 🟡 High | Replace the heuristic with a trained XGBoost/LGBM model. Feature vector: recent form, H2H, ELO ratings, player availability, venue advantage. Train on 2010-2022 World Cup + continental tournament data. |
| **P2.6** | Integrate odds API | 1 day | 🟢 Med | Add The Odds API integration. Display alongside predictions. "Model says 55% Portugal win, bookmakers say 52%." |
| **P2.7** | Add source attribution to responses | 4 hours | 🟢 Med | Show data provenance: "📊 Source: BigQuery (mart_team_form)" or "🌐 Source: Web search (DuckDuckGo)". Builds user trust. |

### Phase 3 — Polish & Scale (Nice-to-Do, 2-3 weeks)

| # | Improvement | Effort | Impact | Description |
|---|------------|--------|--------|-------------|
| **P3.1** | Modern web frontend (React/Next.js) | 2 weeks | 🟡 High | Replace Streamlit with a production React app. Features: streaming chat, match cards, live score ticker, dark mode, mobile-first. Streamlit is great for prototyping but not for a polished consumer product. |
| **P3.2** | Session persistence | 2 days | 🟢 Med | Store conversation history in Redis/Firestore. Users can resume conversations, see past match queries. |
| **P3.3** | Multi-language support | 3 days | 🟢 Med | World Cup is global. At minimum: English, Spanish, Portuguese, French, Arabic. LLM can handle this natively — just add language detection and prompt routing. |
| **P3.4** | Push notifications | 3 days | 🟢 Med | "Kickoff in 15 min: Portugal vs Morocco. Tap for pre-match summary." Web push or WhatsApp template messages. |
| **P3.5** | Fan engagement: polls, predictions leaderboard | 1 week | 🟢 Low-Med | Let users submit score predictions, vote on "player of the match," see aggregated fan predictions vs model. |
| **P3.6** | Data quality monitoring | 2 days | 🟢 Med | Automated checks: row counts, freshness (last ETL timestamp), null rates. Alert if data is stale during live matches. |
| **P3.7** | Align DATA_CONTRACT.md with actual catalog | 2 hours | 🟢 Low | Update the contract to reflect actual marts/dims/facts. Remove gold views that don't exist. |

### Phase 4 — Long-Term (Future)

| # | Improvement | Effort | Impact |
|---|------------|--------|--------|
| **P4.1** | Computer vision: match clip analysis | Large | 🟢 Low (now) |
| **P4.2** | Voice interface ("Hey, who's winning?") | 1 week | 🟢 Med |
| **P4.3** | Personalized notifications per user's followed teams | 3 days | 🟢 Med |
| **P4.4** | Post-match automated recap generation | 3 days | 🟡 High |

---

## 8. Progress Scorecard

### Current Score: **42/100**

The foundation is solid — the orchestrator, BQ agent, catalog system, and guardrails are well-designed. But the app can only answer ~30% of the target questions today (and many with low accuracy). The biggest gaps are player data, live data freshness, and the missing match_facts routing.

### Improvement Contribution to 100%

| # | Improvement | Current Score | Score After | Delta |
|---|------------|--------------|-------------|-------|
| — | **CURRENT** | **42** | — | — |
| P1.1 | Fix match_facts routing | 42 | **48** | +6 |
| P1.2 | dim_player + fact_player_match_stat | 48 | **60** | +12 |
| P1.3 | fact_lineup | 60 | **66** | +6 |
| P1.4 | mart_player_form | 66 | **70** | +4 |
| P1.5 | Streaming orchestrator | 70 | **73** | +3 |
| P1.6 | Live API fallback for match_facts | 73 | **76** | +3 |
| P1.7 | Eliminate redundant classification | 76 | **77** | +1 |
| P2.1 | Pre-match summary agent | 77 | **85** | +8 |
| P2.2 | Live polling cron (5-min) | 85 | **88** | +3 |
| P2.3 | dim_coach | 88 | **90** | +2 |
| P2.4 | Substitution analysis | 90 | **93** | +3 |
| P2.5 | Real ML prediction model | 93 | **95** | +2 |
| P2.6 | Odds API integration | 95 | **96** | +1 |
| P2.7 | Source attribution | 96 | **96.5** | +0.5 |
| P3.1 | Modern web frontend | 96.5 | **99** | +2.5 |
| P3.2-P3.7 | Session, i18n, notifications, etc. | 99 | **100** | +1 |

### Visual Breakdown by Question Type

```
Question Type                         Current    Target
─────────────────────────────────────────────────────────
Fixtures / Schedule                    ✅ 85%     100%
Team Form / Stats                      ✅ 90%     100%
Standings                              ✅ 85%     100%
H2H Records                            ✅ 80%     100%
Match Events (goals, cards)            ⚠️ 60%      95%
Venue / Stadium Info                   ⚠️ 50%      95%
Weather at Venue                       ⚠️ 30%      90%
Referee Info                           ⚠️ 35%      90%
Lineups                                ❌  0%      95%
Player Stats (per game)                ❌  0%      95%
Player Form (last N games)             ❌  0%      95%
Player to Watch / Key Players          ❌  0%      90%
Substitution Analysis                  ❌  0%      85%
Pre-Match Summary                      ❌  5%      95%
Predictions (model)                    ⚠️ 40%      90%
Odds                                   ❌  0%      85%
News                                   ⚠️ 55%      80%
Sentiment                              ⚠️ 20%      60%
─────────────────────────────────────────────────────────
OVERALL                                42%       100%
```

---

## Summary

The architecture is **well-designed at its core** — the catalog-driven BQ agent, the read-only guardrails, the planner+orchestrator pattern, and the cache-first API strategy are all solid decisions. The problem is **what's missing**, not what's broken.

The single biggest unlock is **player-level data** (`dim_player`, `fact_player_match_stat`, `fact_lineup`, `mart_player_form`). Adding these four tables would raise the score from 42 to 70. The next biggest unlock is **live data freshness** (5-min polling cron + match_facts routing fix), taking it to 88. A pre-match summary agent (+ streaming) takes it to 93+. At that point, the app genuinely serves the "fan watching a game" use case.

The quickest win? **P1.1** — add `"match_facts"` to `planner_agent.py:AVAILABLE_AGENTS`. One line. Unlocks live fixture fallback, weather, and web-enhanced match answers immediately.
