# MISA Intelligence API

FastAPI service for querying company intelligence from Postgres and generating investor engagement briefs.

The system is designed for internal MISA-style workflows where users need fast, reliable answers about company profiles, RHQ status, MENA/Saudi presence, and strategic engagement opportunities.

## What This Project Does

- Answers natural-language company questions through `/api/v1/chat`.
- Performs direct structured company search through `/api/v1/search` without using OpenAI.
- Generates strategic investor engagement dossiers through `/api/v1/engagement/generate`.
- Streams chat and dossier output with Server-Sent Events when requested.
- Curates chat answers with OpenAI for insight, sending only **privacy-filtered** rows (internal comments, reviewer notes and audit fields are stripped) with no-retention; falls back to general knowledge (clearly labelled) when the DB has no match.

This is not a generic RAG chatbot. The primary retrieval layer is Postgres SQL over structured company profile data, with fuzzy company search and guardrails around entity matching.

## Architecture

```text
Client
  -> FastAPI
      -> /api/v1/chat
          -> OpenAI tool routing for filter selection
          -> local Postgres query builder (broad smart search across the DB)
          -> OpenAI curation of insight from privacy-filtered rows
          -> general-knowledge fallback (labelled) when no rows
          -> deterministic local commentary if OpenAI is unavailable

      -> /api/v1/search
          -> structured filters
          -> local Postgres query builder
          -> NDJSON rows

      -> /api/v1/engagement/generate
          -> OpenAI Responses API
          -> web_search_preview
          -> strategic dossier markdown
```

Core files:

| Path | Purpose |
| --- | --- |
| `run.py` | Convenience launcher for local development. |
| `app/main.py` | FastAPI app, CORS, routers, and health endpoint. |
| `app/database.py` | Postgres connection cache, schema hints, query builder, smart search. |
| `app/services/chat_engine.py` | NL-to-SQL routing, entity guardrails, retries, trace logging. |
| `app/services/commentary.py` | Deterministic answer generation from database rows. |
| `app/services/engagement_engine.py` | Engagement dossier generation using OpenAI Responses API and web search. |
| `app/routers/v1/` | API route definitions. |
| `tests/` | Unit tests for commentary, cleaner, guardrails, feedback, resilience, and observability. |

## Requirements

- Python 3.11 or newer recommended
- PostgreSQL database with `public.company_profiles`
- `misa_details` JSON data available in `public.company_profiles`
- OpenAI API key for `/api/v1/chat`
- OpenAI API key with Responses API/web search access for `/api/v1/engagement/generate`

Install Python dependencies:

```bash
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Then configure:

```env
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4o-mini

MISA_ENGAGEMENT_OPENAI_API_KEY=sk-proj-...
MISA_ENGAGEMENT_OPENAI_MODEL=gpt-4o

PG_HOST=localhost
PG_PORT=5432
PG_DB=postgres
PG_USER=postgres
PG_PASSWORD=

MISA_LOG_TURNS=false
MISA_LOG_FILE=
MISA_MAX_HISTORY_USER_TURNS=12

# Chat curation / fallback (all optional)
MISA_CHAT_CURATION=true        # OpenAI curates insight from privacy-filtered rows
MISA_CHAT_FALLBACK=true        # OpenAI general-knowledge answer when DB has no rows
MISA_CHAT_OPENAI_STORE=false   # whether OpenAI may retain curation requests
MISA_CHAT_CURATION_MAX_ROWS=15 # max rows sent to OpenAI per answer
MISA_ADVISORY_MAX_COMPLETION_TOKENS=8000 # token budget for strategic-advisory reports
MISA_ADVISORY_OPENAI_MODEL=gpt-4o        # model for advisory reports (mini-tier writes filler)
MISA_DEEP_CURATION=true                  # analytical-depth answers use the advisory model; false = all-mini (cost saver)
```

Notes:

- `MISA_ENGAGEMENT_OPENAI_API_KEY` falls back to `OPENAI_API_KEY` if left empty.
- `MISA_ENGAGEMENT_OPENAI_MODEL` falls back to `OPENAI_MODEL` if left empty.
- Turn logging records metadata such as duration, outcome, locale, and hashed question ID. It does not need to store raw user questions.

## Run Locally

Use the convenience wrapper:

```bash
python run.py
```

Or run Uvicorn directly:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

## API Endpoints

### Health

```http
GET /health
```

Returns Postgres connectivity status and OpenAI configuration status.

### Chat

```http
POST /api/v1/chat
```

Natural-language company intelligence query. Uses OpenAI to choose structured filters, then executes local SQL and generates local commentary.

Example JSON response mode:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Tell me about Alphabet and its MENA presence",
    "history": [],
    "locale": "en",
    "stream": false
  }'
```

Response shape:

```json
{
  "answer": "...",
  "rows": [],
  "trace": [],
  "error": null
}
```

When `stream` is `true`, the endpoint returns `text/event-stream` events:

- `status`
- `rows`
- `chunk`
- `error`
- `done`

### Search

```http
POST /api/v1/search
```

Direct structured company lookup. OpenAI is not used.

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "filters": {
      "rhq_city": {"op": "=", "value": "Riyadh"}
    },
    "order_by": "revenue_usd",
    "descending": true,
    "limit": 20
  }'
```

Returns `application/x-ndjson`: one company row per line.

Allowed filter columns are defined in `app/database.py` under `SCHEMA_HINTS["company_profiles"]["filterable"]`.

### Engagement Dossier

```http
POST /api/v1/engagement/generate
```

Generates a strategic investor engagement dossier. This endpoint uses the OpenAI Responses API with `web_search_preview`, so it may be slower and more expensive than chat/search.

```bash
curl -X POST http://localhost:8000/api/v1/engagement/generate \
  -H "Content-Type: application/json" \
  -d '{
    "entity": "Alphabet",
    "mode": "quick",
    "context": "Focus on Saudi RHQ potential",
    "stream": false
  }'
```

Supported modes:

- `quick`: short strategic brief
- `full`: extended institutional dossier

You can inspect available modes:

```http
GET /api/v1/engagement/modes
```

## AI and Data Safety

The chat pipeline routes with OpenAI, retrieves locally, and curates with OpenAI under a privacy filter:

- OpenAI receives the user question, limited user-message history, fixed system instructions, and schema/tool hints for **filter selection**.
- SQL is built locally from allowlisted columns and parameterized values.
- Unknown filters and unsupported sort columns are ignored by the query builder.
- The application only builds `SELECT` statements.
- For **curation**, retrieved rows are sent to OpenAI to compose insight, but only after a privacy filter strips internal/sensitive fields (team/MISA/reviewer comments, review status, creator/audit metadata, external-source paths). Rows are capped (`MISA_CHAT_CURATION_MAX_ROWS`) and long text is truncated, and requests are sent with `store=false` by default (no retention).
- When the DB returns no rows, OpenAI may answer from general knowledge; such answers are clearly labelled as **not** sourced from the MISA database.
- If OpenAI is unavailable or errors, the pipeline degrades to deterministic local commentary generated from rows (no second LLM call). Set `MISA_CHAT_CURATION=false` to force this mode and keep all row data local.

The engagement dossier endpoint is different: it uses OpenAI web search to produce a research brief. Do not send confidential internal context to that endpoint unless your deployment policy allows it.

## Retrieval Strategy

This project does not use embeddings or vector search today. That is a good trade-off for the current structured-data use case:

- Lower infrastructure complexity
- Lower retrieval cost
- More explainable SQL traces
- Better privacy boundary for row-level company data

Consider adding hybrid search, reranking, or embeddings only if users need semantic search over long unstructured fields such as notes, PDFs, transcripts, or meeting summaries.

## Tests

Run all tests:

```bash
pytest -v
```

Run a focused test file:

```bash
pytest tests/test_pipeline_guardrails.py -v
```

## Performance and Cost Notes

- `/api/v1/search` is the cheapest path because it does not call OpenAI.
- `/api/v1/chat` has one or more OpenAI Chat Completions calls depending on retries and validation.
- `/api/v1/engagement/generate` is the highest-latency and highest-cost path because it can perform live web search and generate long dossiers.
- Postgres connections are cached at module level and retried on transient connection failures.
- Query limits are capped in the database layer to reduce accidental large reads.

## Production Gaps

Before exposing this outside a trusted internal environment, add:

- Authentication and authorization
- Role-aware field redaction or database row-level security
- Request rate limiting
- Full audit logging for compliance-sensitive usage
- CORS restrictions for approved origins only
- Secrets management outside local `.env`
- Deployment health checks and metrics

## Documentation

Additional architecture notes live in:

- `docs/ARCHITECTURE_AND_SQL_PIPELINE.md`

That document may still reference older Streamlit-era filenames in places. Treat the current `app/` source code as authoritative when behavior differs.
