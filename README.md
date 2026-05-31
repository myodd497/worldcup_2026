# World Cup 2026 — AI Insights Platform

> **Deadline: June 10, 2026 (11 days). World Cup 2026 kicks off June 11.**

---

## Project Vision

A multi-agent AI system reachable via WhatsApp that answers questions about World Cup 2026 matches in real time. An orchestrator agent receives messages and routes tasks to a pool of specialized agents — news retrieval, sentiment analysis, match facts, prediction, documentation, code review, and data warehousing.

---

## System Architecture

## v2 Architecture (Router + Confidence + Result Composer)

```
┌────────────────────────────────────────────────────────────────────────────┐
│ USER (WhatsApp)                                                            │
│   → Twilio Webhook                                                         │
│   → FastAPI /webhook                                                       │
│   → LangGraph Orchestrator                                                 │
│                                                                            │
│ LangGraph runtime path                                                     │
│   classify_intent  →  router  →  specialist_agent  →  confidence  → compose│
│                                                                            │
│ Specialist agents (single selected route per request)                      │
│   news | sentiment | match_facts | prediction | other                      │
│                                                                            │
│ compose node                                                                │
│   - builds final WhatsApp response                                          │
│   - appends confidence label + reason (when needed)                         │
│   - persists session log via Docs Agent                                     │
│                                                                            │
│ Data and model layer                                                        │
│   BigQuery (raw + curated + analytical tables)                              │
│   MLflow + model artifacts                                                  │
└────────────────────────────────────────────────────────────────────────────┘
```

## LangGraph Node Flow (Exact)

### State Fields

The orchestrator state now carries these fields:

- `user_id: str`
- `user_message: str`
- `intent: str`
- `selected_agent: str`
- `agent_payload: dict[str, Any]`
- `confidence_score: float`
- `confidence_label: str`
- `confidence_reason: str`
- `final_reply: str`
- `messages: list`

### Node Sequence

1. `classify`
Classifies the user message into one of: `news`, `sentiment`, `match_facts`, `prediction`, `other`.

2. `router`
Maps intent to one selected specialist node (`selected_agent`).

3. Specialist node
Runs exactly one of:
- `news`
- `sentiment`
- `match_facts`
- `prediction`
- `other`

Each specialist now returns structured payload:
- `answer`
- `confidence_score`
- `confidence_reason`
- `metadata`

4. `confidence`
Normalizes score into `[0.0, 1.0]` and labels:
- `high` if score `>= 0.80`
- `medium` if `0.55 <= score < 0.80`
- `low` if score `< 0.55`

5. `compose`
Uses the Result Composer Agent to produce final user response.
For low confidence, includes reason and a refinement tip.
Also logs the final response through Docs Agent.

### Edges

The graph edges are:

- `classify -> router`
- `router -> news | sentiment | match_facts | prediction | other`
- `(news | sentiment | match_facts | prediction | other) -> confidence`
- `confidence -> compose`
- `compose -> END`

---

## Agent Responsibilities

| Agent | Role | Primary Tools |
|---|---|---|
| **Orchestrator** | Parses user intent, routes to specialist agents, composes final reply | LangGraph, GPT-4o |
| **News Agent** | Fetches latest news about a match/team/player | Tavily, NewsAPI |
| **Sentiment Agent** | Analyses social media buzz around a game | Twitter/X API v2, VADER, DistilBERT |
| **Match Facts Agent** | Returns lineups, venue, weather, referee, standings | API-Football, OpenWeatherMap |
| **Prediction Agent** | Generates match outcome probabilities | scikit-learn (XGBoost), LLM chain-of-thought |
| **Result Composer Agent** | Builds final WhatsApp response from structured outputs and confidence signals | LangGraph node + formatting logic |
| **BigQuery Agent** | Uploads processed data to BQ, runs analytical queries | google-cloud-bigquery |
| **Docs Agent** | Writes session summaries, keeps docs up-to-date | LLM + file tools |
| **Code Review Agent** | Reviews generated Python snippets before execution | Ruff, mypy, LLM review |

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

## Repository Structure (target)

```
worldcup_2026/
├── pyproject.toml
├── README.md
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── src/
│   ├── __init__.py
│   ├── entrypoint.ipynb
│   ├── prepare_workspace.py
│   ├── server/                     # FastAPI webhook
│   │   ├── __init__.py
│   │   ├── app.py
│   │   └── whatsapp_handler.py
│   ├── agents/                     # All agents
│   │   ├── __init__.py
│   │   ├── orchestrator.py
│   │   ├── result_composer_agent.py
│   │   ├── news_agent.py
│   │   ├── sentiment_agent.py
│   │   ├── match_facts_agent.py
│   │   ├── prediction_agent.py
│   │   ├── bigquery_agent.py
│   │   ├── docs_agent.py
│   │   └── code_review_agent.py
│   ├── tools/                      # Reusable tool functions for agents
│   │   ├── __init__.py
│   │   ├── api_football.py
│   │   ├── weather.py
│   │   ├── news_search.py
│   │   ├── twitter_sentiment.py
│   │   └── bigquery_tools.py
│   ├── models/                     # ML prediction models
│   │   ├── __init__.py
│   │   ├── feature_engineering.py
│   │   ├── train.py
│   │   └── predict.py
│   └── data/                       # Data ingestion & processing
│       ├── __init__.py
│       ├── ingest_historical.py
│       └── schemas.py
└── bin/
    ├── artifacts/
    ├── data_outputs/
    ├── docs/
    ├── mlruns/
    ├── models_deployed/
    └── scripts/
        ├── run_server.sh
        └── deploy_cloud_run.sh
```

---

## 11-Day Sprint Plan

> Start: **May 30, 2026** → Deadline: **June 10, 2026**

### Day 1–2 | Infrastructure & Environment
**Goal:** Everything boots, secrets are wired, skeleton modules exist.

- [ ] Update `pyproject.toml` — add all new dependencies
- [ ] Create `.env.example` with all required API keys
- [ ] Create `Dockerfile` + `docker-compose.yml`
- [ ] Scaffold all `src/` directories and `__init__.py` files
- [ ] Create BigQuery dataset + service account JSON
- [ ] Test Twilio WhatsApp sandbox (send/receive one message)
- [ ] Set up MLflow tracking server (local, `bin/mlruns/`)

**Deliverable:** `docker-compose up` runs without error; WhatsApp sandbox replies "pong".

---

### Day 3–4 | Data Ingestion & Tools Layer
**Goal:** All external APIs callable, data flowing into BigQuery.

- [ ] `src/tools/api_football.py` — matches, lineups, standings, fixtures
- [ ] `src/tools/weather.py` — venue city forecast
- [ ] `src/tools/news_search.py` — Tavily + NewsAPI integration
- [ ] `src/tools/twitter_sentiment.py` — recent tweets, VADER scoring
- [ ] `src/tools/bigquery_tools.py` — upload DataFrame, run SQL query
- [ ] `src/data/ingest_historical.py` — fetch historical WC data → BigQuery
- [ ] `src/data/schemas.py` — Pydantic models for all data contracts
- [ ] Notebook: `src/entrypoint.ipynb` — exploratory validation of all tools

**Deliverable:** Each tool has a standalone `if __name__ == "__main__"` test. All return valid data.

---

### Day 5–6 | Specialist Agents
**Goal:** Each agent callable in isolation with a test question.

- [ ] `src/agents/news_agent.py` — LangGraph node, wraps `news_search` tool
- [ ] `src/agents/sentiment_agent.py` — wraps `twitter_sentiment` + DistilBERT classifier
- [ ] `src/agents/match_facts_agent.py` — wraps `api_football` + `weather` tools
- [ ] `src/agents/bigquery_agent.py` — wraps `bigquery_tools`, exposes upload + query actions
- [ ] `src/agents/docs_agent.py` — writes session log markdown to `bin/docs/`
- [ ] `src/agents/code_review_agent.py` — runs Ruff + mypy, then LLM review pass
- [ ] Unit tests in `tests/test_agents.py`

**Deliverable:** `pytest tests/test_agents.py` — all agents return structured output for a test match.

---

### Day 7–8 | Prediction Model
**Goal:** A trained model that returns win/draw/loss probabilities.

- [ ] `src/data/ingest_historical.py` — finalize historical data (FIFA rankings, ELO, form, H2H)
- [ ] `src/models/feature_engineering.py` — build feature matrix
- [ ] `src/models/train.py` — XGBoost classifier, MLflow experiment logged
- [ ] `src/models/predict.py` — load model from `bin/models_deployed/`, return probabilities
- [ ] `src/agents/prediction_agent.py` — wraps `predict.py` + LLM chain-of-thought explanation
- [ ] Notebook: validate model accuracy on test split, log to MLflow

**Deliverable:** Model serialised to `bin/models_deployed/wc2026_predictor.pkl`. Prediction agent returns: "Portugal 58% | Draw 22% | Morocco 20%".

---

### Day 9 | Orchestrator + WhatsApp Server
**Goal:** Full message round-trip from WhatsApp → orchestrator → specialist agents → reply.

- [ ] `src/agents/orchestrator.py` — LangGraph `StateGraph`, intent classification, agent routing
- [ ] `src/server/app.py` — FastAPI app, `/webhook` POST endpoint
- [ ] `src/server/whatsapp_handler.py` — Twilio signature validation, message parsing, reply sending
- [ ] Wire orchestrator into webhook handler
- [ ] End-to-end test: "What are the lineups for Portugal vs Morocco?" → full reply

**Deliverable:** Running locally with `ngrok`, real WhatsApp message answered correctly.

---

### Day 10 | Integration, BigQuery Pipeline & Polish
**Goal:** Data flows to BigQuery; all agents co-operate.

- [ ] BigQuery agent uploads match facts + predictions after each query
- [ ] Docs agent logs every session to `bin/docs/`
- [ ] Code review agent gates any dynamically generated code before execution
- [ ] Add rate-limiting to FastAPI (per WhatsApp number)
- [ ] Secrets management review — no keys in code
- [ ] `bin/scripts/run_server.sh` — single command to start everything

**Deliverable:** A full demo flow with 5 different question types all working end-to-end.

---

### Day 11 | Deployment & Final Testing
**Goal:** Live on the internet, stable, documented.

- [ ] `Dockerfile` + `bin/scripts/deploy_cloud_run.sh` — deploy to Cloud Run (or Railway as fallback)
- [ ] Set production Twilio webhook to Cloud Run URL
- [ ] Environment variables in Cloud Run secrets
- [ ] Smoke test all 7 agent types from real WhatsApp
- [ ] Final README pass — update with actual API key setup steps
- [ ] Tag release `v1.0.0`

**Deliverable:** Public URL, real WhatsApp number answering World Cup questions.

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
LangGraph gives explicit control over agent state and message routing — critical when WhatsApp sessions must be stateful across multiple turns. CrewAI is higher-level but less controllable for production webhook flows.

### Why XGBoost over a pure LLM for predictions?
LLMs hallucinate probabilities. A trained XGBoost on historical match data (ELO ratings, FIFA rankings, form, H2H records, tournament stage) gives calibrated probabilities. The LLM then **explains** those probabilities in natural language.

### Why Cloud Run for hosting?
Serverless, scales to zero (low cost during quiet periods), handles Twilio webhook latency requirements, integrates natively with BigQuery and Secret Manager.

### Why BigQuery as the sole data store?
All data — raw fixtures, match events, predictions, session logs — flows directly into BigQuery. This keeps the stack simple: one query engine, one IAM model, and SQL for everything. BigQuery handles both the operational queries the agents make in real time and the analytical queries for model training, with no extra infrastructure to maintain.

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