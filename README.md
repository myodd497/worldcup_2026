# World Cup 2026 — AI Insights Platform

> **Deadline: June 10, 2026 (11 days). World Cup 2026 kicks off June 11.**

---

## Project Vision

A multi-agent AI system reachable via WhatsApp that answers questions about World Cup 2026 matches in real time. An orchestrator agent receives messages and routes tasks to a pool of specialized agents — news retrieval, sentiment analysis, match facts, prediction, documentation, code review, and data warehousing.

---

## System Architecture

## v2 Architecture (Multi-Agent + Confidence + Result Composer)

```
┌────────────────────────────────────────────────────────────────────────────┐
│ USER (WhatsApp / Streamlit / Web)                                          │
│   → Twilio Webhook  /  Streamlit Chat  /  FastAPI                          │
│   → LangGraph Orchestrator                                                 │
│                                                                            │
│ LangGraph runtime path (4-node pipeline)                                    │
│   plan (gpt-4o)  →  execute (1-2 specialists)  →  verify (gpt-4o critic)   │
│                  →  compose (gpt-4o-mini, WhatsApp-friendly)                │
│                                                                            │
│ Specialist agents                                                          │
│   bigquery | prediction | news | sentiment | rules | chat                  │
│                                                                            │
│ State-of-the-art BQ stack (the quality-critical path)                      │
│   - entity_resolver (deterministic team/player → id)                       │
│   - schema_retriever (top-K relevant tables only)                          │
│   - sql_few_shots (retrieved Q→SQL examples)                                │
│   - validate → dry-run + cost guard → execute → repair loop (max 2)         │
│   - verifier_agent (LLM-as-judge) can trigger ONE repair attempt            │
│                                                                            │
│ Conversation memory                                                        │
│   ConversationMemory: rolling LLM summary + structured entity store        │
│                                                                            │
│ Data layer                                                                 │
│   BigQuery (catalog-driven, see src/data/datamodel/catalog.py)             │
└────────────────────────────────────────────────────────────────────────────┘
```

## LangGraph Node Flow (Exact)

### State Fields

The orchestrator state carries these fields:

- `user_id: str`
- `user_message: str`
- `conversation_context: str` — rolling summary + entity store + last raw turns
- `topic: str` — short noun phrase from the planner
- `selected_agent: str` — primary agent for compose
- `selected_agents: list[str]` — 1-2 agents chosen by planner
- `needs_verifier: bool` — planner-set flag for the verifier step
- `agent_outputs: dict[str, dict]` — per-agent structured results
- `agent_payload: dict` — primary payload that flows to compose
- `verifier_verdict: dict` — critic's findings (groundedness, issues, repair hint)
- `confidence_score: float`, `confidence_label: str`, `confidence_reason: str`
- `final_reply: str`

### Node Sequence (4 nodes)

1. **`plan`** — one `gpt-4o` call returns `{agents, primary_agent, topic, needs_verifier, reason}`. Replaces the old classify + route two-step.
2. **`execute`** — runs the selected specialists sequentially. Each returns `{answer, confidence_score, confidence_reason, metadata}`.
3. **`verify`** — when `needs_verifier` is set and the primary is BigQuery, a `gpt-4o` critic scores groundedness and may request ONE structured repair from the BigQuery agent.
4. **`compose`** — `gpt-4o-mini` formats the final WhatsApp-friendly reply with ⭐ confidence line and (for low confidence) a refinement tip.

### Edges

```
plan → execute → verify → compose → END
```

---

## Agent Responsibilities

| Agent | Role | Primary Tools |
|---|---|---|
| **Orchestrator** | Parses user intent, routes to specialist agents (1–3), aggregates multi-agent outputs, composes final reply | LangGraph, GPT-4o-mini, WorkflowTracker |
| **Planner Agent** | Selects 1–3 specialist agents for a query, with keyword fallback for robustness | GPT-4o-mini, structured JSON output |
| **News Agent** | Fetches latest news about a match/team/player (5-provider fallback: DuckDuckGo → Serper → Tavily → NewsAPI → Google News RSS) | Tavily, NewsAPI, DuckDuckGo HTML |
| **Sentiment Agent** | Analyses social media buzz around a game | Twitter/X API v2, VADER |
| **Match Facts Agent** | Returns fixtures, lineups, venue, weather, referee, standings (cache-first: BQ → API-Football → web → LLM) | API-Football, OpenWeatherMap, web scraping, BigQuery |
| **Prediction Agent** | Generates match outcome probabilities (cache-first: BQ heuristic → API warm → uniform fallback) | BigQuery, API-Football, XGBoost (planned) |
| **BigQuery Agent** | Catalog-driven BQ querying via function calling (list → describe → sample → run_sql → format). Single source of truth for all structured football data | google-cloud-bigquery, datamodel catalog |
| **Rules Agent** | Answers FIFA World Cup 2026 regulations questions (Articles 1–52). Cites specific articles from official document | GPT-4o-mini, `Docs/FWC26_regulations_EN.txt` |
| **Result Composer Agent** | Formats agent output for end-user display. Adds confidence line, low-confidence tips, prediction cautions | GPT-4o-mini, formatting logic |
| **Docs Agent** | Writes session summaries, keeps docs up-to-date | LLM + file tools |
| **Code Review Agent** | Reviews generated Python snippets before execution | Ruff, mypy, LLM review |
| **Workflow Logger** | Tracks orchestrator node execution with timestamps, input/output snapshots, full JSON trace | In-memory tracker, session-scoped |

---

## Technology Stack

### Core Framework
| Purpose | Library | Version |
|---|---|---|
| Agent orchestration | `langgraph` | ≥0.2 |
| LLM clients | `langchain-openai` / `langchain-anthropic` | latest |
| Web server / webhook | `fastapi` + `uvicorn` | latest |
| WhatsApp channel | `twilio` | ≥9.0 |
| Task queue (optional) | `celery` + `redis` | latest |

### Data & APIs
| Purpose | Library / API |
|---|---|
| Sports data | `API-Football` (rapidapi.com/api-sports) |
| Weather | `openweathermap` API |
| News search | `tavily-python`, `newsapi-python` |
| Social sentiment | `tweepy` (Twitter/X API v2) |
| Sentiment model | `vaderSentiment`, `transformers` (DistilBERT) |
| BigQuery | `google-cloud-bigquery`, `db-dtypes` |

### ML / Modelling
| Purpose | Library |
|---|---|
| Feature engineering | `pandas`, `numpy` (already present) |
| Prediction models | `scikit-learn`, `xgboost` |
| Experiment tracking | `mlflow` (already present) |
| Model serialisation | `joblib` |

### Dev & Quality
| Purpose | Library |
|---|---|
| Linting | `ruff` |
| Type checking | `mypy` |
| Testing | `pytest`, `pytest-asyncio` |
| Env management | `python-dotenv` (already present) |
| Containerisation | `Docker` + `docker-compose` |

---

## Repository Structure (current)

```
worldcup_2026/
├── pyproject.toml
├── README.md
├── AGENT_SYSTEM_ANALYSIS.md
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── Docs/
│   └── FWC26_regulations_EN.txt      # Official FIFA regulations (Articles 1–52)
├── src/
│   ├── __init__.py
│   ├── entrypoint.ipynb
│   ├── prepare_workspace.py
│   ├── workflow_testing_notebook.ipynb
│   ├── server/                     # FastAPI + Streamlit interfaces
│   │   ├── __init__.py
│   │   ├── app.py                  # FastAPI webhook server
│   │   ├── streamlit_app.py        # Streamlit chat UI (alternative)
│   │   └── whatsapp_handler.py     # Twilio message handling
│   └── agents/                     # Orchestrator + planner + specialists
│       ├── __init__.py
│       ├── orchestrator.py         # LangGraph 4-node pipeline
│       ├── planner_agent.py        # gpt-4o single-call router
│       ├── conversation_memory.py  # Rolling summary + entity store
│       ├── verifier_agent.py       # LLM-as-judge critic
│       ├── result_composer_agent.py
│       ├── bigquery_agent.py       # Retrieval-driven SQL agent (gpt-4o)
│       ├── sql_few_shots.py        # Q→SQL example library
│       ├── news_agent.py
│       ├── sentiment_agent.py
│       ├── prediction_agent.py     # BQ-only heuristic + planned ML
│       ├── rules_agent.py
│       ├── docs_agent.py
│       ├── code_review_agent.py
│       └── workflow_logger.py
│   ├── tools/                      # Reusable tool functions for agents
│   │   ├── __init__.py
│   │   ├── entity_resolver.py       # Deterministic team/player resolution
│   │   ├── weather.py              # OpenWeatherMap client
│   │   ├── news_search.py          # Multi-provider news search
│   │   ├── twitter_sentiment.py    # Twitter/X sentiment
│   │   ├── bigquery_tools.py       # BQ query execution
│   │   ├── datamodel_tools.py      # Catalog-driven BQ tools (list/describe/sample/run_sql)
│   │   └── api_usage_tracker.py    # API call tracking
│   ├── models/                     # ML prediction models
│   │   ├── __init__.py
│   │   ├── feature_engineering.py
│   │   ├── train.py
│   │   └── predict.py
│   └── data/                       # Data ingestion & BigQuery data model
│       ├── __init__.py
│       ├── startup_etl.py
│       └── datamodel/              # 20 tables: 4 dims + 4 facts + 6 marts + 4 raw + catalog
│           ├── __init__.py
│           ├── catalog.py          # Self-documenting table metadata
│           ├── build_datamodel.py
│           ├── dim_team.py, dim_competition.py, dim_venue.py, dim_date.py
│           ├── fact_match.py, fact_match_team.py, fact_match_event.py, fact_standings_snapshot.py
│           ├── mart_team_profile.py, mart_team_form.py, mart_head_to_head.py
│           ├── mart_match_history.py, mart_match_upcoming.py, mart_tournament_state.py
│           └── raw_fixtures.py, raw_fixture_events.py, raw_fixture_statistics.py, raw_standings.py
├── bin/
│   ├── artifacts/
│   ├── data_outputs/
│   ├── docs/                       # Session logs
│   ├── mlruns/
│   ├── models_deployed/
│   └── scripts/
│       ├── run_server.sh
│       └── deploy_cloud_run.sh
├── tests/
│   └── test_agents.py
└── secrets/
    └── gcp_service_account.json
```

---

## Sprint Progress (May 30 — June 10, 2026)

> Current date: **June 7, 2026** — Day 9 of 11. **3 days to deadline.**

### ✅ Done

| Day | Scope | Status |
|-----|-------|--------|
| 1–2 | Infrastructure: `pyproject.toml`, `Dockerfile`, `docker-compose.yml`, BQ dataset, service account, `.env.example` | ✅ Complete |
| 3–4 | Tools layer: `weather.py`, `news_search.py`, `twitter_sentiment.py`, `bigquery_tools.py`, `datamodel_tools.py`, `entity_resolver.py`, `api_usage_tracker.py` | ✅ Complete |
| 3–4 | Data model: 20-table catalog (4 dims + 4 facts + 6 marts + 4 raw), `catalog.py`, `build_datamodel.py`, `startup_etl.py` | ✅ Complete |
| 5–6 | Specialist agents: `news_agent`, `sentiment_agent`, `bigquery_agent`, `prediction_agent`, `rules_agent`, `docs_agent`, `code_review_agent`, `verifier_agent` | ✅ Complete |
| 5–6 (extra) | New: `rules_agent.py` — FIFA regulations Q&A from official document | ✅ Complete |
| 5–6 (extra) | New: `planner_agent.py` — agent selection with keyword fallback | ✅ Complete |
| 5–6 (extra) | New: `workflow_logger.py` — execution tracing | ✅ Complete |
| 9 | Orchestrator: 4-node LangGraph pipeline (plan / execute / verify / compose) with rolling-summary memory and verifier-driven repair | ✅ Complete |
| 9 | Server: `app.py` (FastAPI), `streamlit_app.py` (web UI), `whatsapp_handler.py` (Twilio) | ✅ Complete |

### ⚠️ In Progress / Partial

| Day | Scope | Status |
|-----|-------|--------|
| 7–8 | Prediction model: heuristic works, XGBoost path is TODO | ⚠️ Heuristic only |
| 9 | End-to-end WhatsApp test | ⚠️ Needs ngrok setup |
| 10 | Docs agent session logging wired | ⚠️ In orchestrator but not fully tested |

### ❌ Remaining

| Day | Scope | Priority |
|-----|-------|----------|
| 10 | Rate-limiting, secrets audit, `run_server.sh` | 🟡 High |
| 10 | BigQuery agent uploads after each query | 🟡 High |
| 11 | Cloud Run deployment (`deploy_cloud_run.sh`) | 🟡 High |
| 11 | Production Twilio webhook + smoke test | 🟡 High |
| 7–8 | Real XGBoost model training (replaces heuristic) | 🟢 Med |

---

## API Keys & Secrets Required

Create a `.env` file (never commit it — it is in `.gitignore`):

```env
# LLM
OPENAI_API_KEY=

# WhatsApp / Twilio
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886

# Sports
API_FOOTBALL_KEY=           # rapidapi.com → api-sports/api-football

# Weather
OPENWEATHER_API_KEY=        # openweathermap.org

# News
TAVILY_API_KEY=             # tavily.com
NEWSAPI_KEY=                # newsapi.org

# Social
TWITTER_BEARER_TOKEN=       # developer.twitter.com

# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=bin/secrets/gcp_service_account.json
BIGQUERY_PROJECT_ID=
BIGQUERY_DATASET_ID=worldcup2026
```

---

## Key Design Decisions

### Why LangGraph over CrewAI / AutoGen?
LangGraph gives explicit control over agent state and message routing — critical when WhatsApp sessions must be stateful across multiple turns. CrewAI is higher-level but less controllable for production webhook flows. The current 4-node pipeline (`plan → execute → verify → compose`) provides full observability and a verifier-driven repair loop.

### Why a Planner Agent separate from the Orchestrator?
The Planner (`planner_agent.py`) handles agent selection with a structured JSON contract and keyword-based fallback. Separating this from the orchestrator keeps each component focused: the orchestrator manages state flow, the planner decides **who** to call. This also makes the planner independently testable.

### Why catalog-driven BQ agent design?
The BQ agent discovers tables dynamically via `catalog.py` (`list_tables → describe_table → sample_table → run_sql → format`). No table names are hardcoded. This makes the system self-documenting, adaptable to schema changes, and enforces read-only access via table allow-listing in `datamodel_tools.py`.

### Why cache-first for API calls?
`prediction_agent` reads exclusively from BigQuery (the gold model is the single source of truth at runtime). Live API ingestion is owned by the ETL scheduler, not the chat path.

### Why XGBoost over a pure LLM for predictions?
LLMs hallucinate probabilities. A trained XGBoost on historical match data (ELO ratings, FIFA rankings, form, H2H records, tournament stage) gives calibrated probabilities. The LLM then **explains** those probabilities in natural language. Currently the heuristic provides directional guidance while the XGBoost model is being trained.

### Why Cloud Run for hosting?
Serverless, scales to zero (low cost during quiet periods), handles Twilio webhook latency requirements, integrates natively with BigQuery and Secret Manager.

### Why BigQuery as the sole data store?
All data — raw fixtures, match events, predictions, session logs — flows directly into BigQuery. This keeps the stack simple: one query engine, one IAM model, and SQL for everything. The 20-table catalog (4 dims + 4 facts + 6 marts + 4 raw) provides a complete star schema for the World Cup domain.

---

## Setup Instructions

### macOS

1. **Install Poetry**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   
   After installation, add Poetry to your PATH by adding this line to your shell profile (`~/.zshrc` or `~/.bash_profile`):
   ```bash
   export PATH="$HOME/.local/bin:$PATH"
   ```
   
   Then reload your shell:
   ```bash
   source ~/.zshrc
   ```

2. **Configure Poetry to use in-project virtual environments**
   ```bash
   poetry config virtualenvs.in-project true
   ```

3. **Install project dependencies**
   ```bash
   poetry install
   ```

### Windows

1. **Install Poetry**
   
   Using PowerShell:
   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
   ```
   
   Or using Windows Package Manager:
   ```powershell
   winget install Python.Poetry
   ```

2. **Configure Poetry to use in-project virtual environments**
   ```cmd
   poetry config virtualenvs.in-project true
   ```

3. **Install project dependencies**
   ```cmd
   poetry install
   ```
