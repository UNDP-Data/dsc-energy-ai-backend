# AI API Operations

## Local setup

From repository README and Makefile:

```bash
cd /Users/ben/Documents/UNDP/SEH/dsc-energy-ai-backend
python -m venv .venv
source .venv/bin/activate
make install
```

The application reads `.env` through `python-dotenv` during import.

## Running the API locally

```bash
cd /Users/ben/Documents/UNDP/SEH/dsc-energy-ai-backend
source .venv/bin/activate
make run
```

Equivalent command:

```bash
uvicorn main:app --reload
```

Default local API URL:

```text
http://127.0.0.1:8000
```

## Running the local tester

The standalone tester app is separate from the main API process.

```bash
cd /Users/ben/Documents/UNDP/SEH/dsc-energy-ai-backend
source .venv/bin/activate
make run-tester
```

Tester URL:

```text
http://127.0.0.1:8010/kg-tester
```

Notes:

- The tester proxies to the configured local or remote API.
- It reads `KG_TESTER_API_KEY` first, then `API_KEY`.
- It is loopback-only.

## Running tests

```bash
cd /Users/ben/Documents/UNDP/SEH/dsc-energy-ai-backend
source .venv/bin/activate
python -m pytest tests
```

The full suite can take several minutes because some tests exercise model and retrieval behavior.

## Ingestion and corpus metadata commands

Build `sources` and `documents` from the existing `chunks` table without rewriting chunks:

```bash
python3 scripts/bootstrap_corpus_tables.py --overwrite
```

Rewrite chunks with document provenance fields when the environment and storage allow it:

```bash
python3 scripts/bootstrap_corpus_tables.py --overwrite --rewrite-chunks
```

Validate corpus tables:

```bash
python3 scripts/validate_corpus_tables.py
```

Import a manifest:

```bash
python3 scripts/import_corpus_manifest.py --manifest data/corpus/sample_manifest.yaml --include-chunks
```

Replace the governed SGP intranet-project partition while preserving the
Innovation Library partition:

```bash
python3 scripts/deploy_corpus_partition.py \
  --manifest /path/to/rag-export/corpus/manifest.yaml

python3 scripts/deploy_corpus_partition.py \
  --manifest /path/to/rag-export/corpus/manifest.yaml \
  --apply \
  --allow-non-public-external-embedding
```

The first command is a read-only plan. The applied command embeds missing
chunks, validates isolated staging tables and per-source counts, and only then
swaps the live tables. It writes a deployment audit beside the manifest.
The explicit external-embedding flag confirms approval to send non-public or
sensitive-flagged chunk text to the configured embedding provider.

Export document inventory:

```bash
python3 scripts/export_documents_inventory.py --output data/retrieval_benchmark/corpus_inventory.csv
```

Unknown / requires confirmation:

- The official production process for rebuilding `chunks` from raw publications.
- Whether `main.ipynb` is still the authoritative ingestion notebook.
- Whether the team should rewrite production chunks directly or create a new versioned LanceDB location first.

## Retrieval and model debugging commands

Run retrieval surface test prompts and save logs:

```bash
python3 scripts/run_retrieval_surface.py --base-url http://127.0.0.1:8000 --output-prefix tmp/retrieval_surface
```

Run streaming model debug prompts and save raw output:

```bash
python3 scripts/debug_model_stream.py --base-url http://127.0.0.1:8000 --output tmp/model_stream_debug.txt
```

Query retrieval debug endpoint:

```bash
curl -sS -H "X-Api-Key: $API_KEY" "http://127.0.0.1:8000/debug/retrieve?query=how%20many%20people%20lack%20access%20to%20energy"
```

Inspect table health:

```bash
curl -sS -H "X-Api-Key: $API_KEY" "http://127.0.0.1:8000/debug/tables"
```

## Knowledge graph evaluation commands

Evaluate graph v2 with default query set:

```bash
python3 scripts/evaluate_graph_v2.py --base-url http://127.0.0.1:8000 --output tmp/graph_v2_eval.json
```

Unknown / requires confirmation:

- Whether graph evaluation is part of production release gating.

## Deployment assumptions

Confirmed from `.github/workflows/azure-webapps-python.yml`:

- Deployment runs on push to `main`.
- The target Azure Web App is `sea-ai-api`.
- The workflow uses Python 3.12.
- Dependencies are installed from `requirements.txt`.
- Deployment uses `azure/webapps-deploy@v2` and publish profile secret `AZURE_WEBAPP_PUBLISH_PROFILE`.

Confirmed from `Dockerfile`:

- Docker runtime is `python:3.12-slim`.
- It exposes port `8000`.
- It runs `uvicorn main:app --host 0.0.0.0 --port 8000`.

Unknown / requires confirmation:

- Whether production currently uses the Dockerfile or Azure App Service build output.
- Production startup command in Azure App Service.
- Production environment-variable source and secret rotation process.

## Required environment variables

Core API:

- `API_KEY`
- `AZURE_OPENAI_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_CHAT_MODEL`
- `AZURE_OPENAI_EMBED_MODEL`

Storage:

- `STORAGE_SAS_URL`, or
- `STORAGE_ACCOUNT_NAME` plus one of:
  - `STORAGE_ACCOUNT_KEY`
  - `STORAGE_SAS_TOKEN`

Optional model/runtime tuning:

- `AZURE_OPENAI_TIMEOUT`
- `MODEL_SCOPE_GUARD_ENABLED`
- `MODEL_GRAPH_TIMEOUT_SECONDS`
- `MODEL_PUBLICATION_RETRIEVAL_TIMEOUT_SECONDS`
- `MODEL_STREAM_IDLE_TIMEOUT_SECONDS`
- `MODEL_TOOLS_PREP_TIMEOUT_SECONDS`
- `MODEL_STREAM_WATCHDOG_SECONDS`
- `MODEL_PUBLICATION_HEARTBEAT_SECONDS`
- `RETRIEVE_CHUNKS_LEXICAL_TIMEOUT_SECONDS`
- `RETRIEVE_CHUNKS_VARIANT_TIMEOUT_SECONDS`
- `RETRIEVE_DOCUMENT_INDEX_TIMEOUT_SECONDS`
- `RETRIEVE_CURRENT_DATA_ENRICHMENT_TIMEOUT_SECONDS`
- `DOCUMENT_INDEX_CACHE_TTL_SECONDS`
- `LANCE_WARN_UNSAFE_RETRY`

Moonshot:

- `ALLOWED_ORIGINS`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_ORG_ID`
- `OPENAI_PROJECT_ID`
- `OPENAI_PARSE_MODEL`
- `OPENAI_SYNOPSIS_MODEL`
- `AZURE_OPENAI_MOONSHOT_PARSE_DEPLOYMENT`
- `AZURE_OPENAI_MOONSHOT_SYNOPSIS_DEPLOYMENT`
- `AZURE_OPENAI_PARSE_DEPLOYMENT`
- `AZURE_OPENAI_SYNOPSIS_DEPLOYMENT`
- `AZURE_OPENAI_MOONSHOT_API_VERSION`
- `MOONSHOT_RATE_LIMIT_WINDOW_SECONDS`
- `MOONSHOT_RATE_LIMIT_PARSE_MAX`
- `MOONSHOT_RATE_LIMIT_SYNOPSIS_MAX`

Local tester:

- `KG_TESTER_API_KEY`
- `API_KEY`
- Additional tester-specific target URL environment variables are Unknown / requires confirmation from runtime usage.

## Common failures and troubleshooting

### Missing storage credentials

Symptom:

```text
RuntimeError: Missing `STORAGE_ACCOUNT_NAME` environment variable. Populate it in your `.env` file.
```

Resolution:

- Provide `STORAGE_SAS_URL`, or provide `STORAGE_ACCOUNT_NAME` and either `STORAGE_ACCOUNT_KEY` or `STORAGE_SAS_TOKEN`.

### Missing or invalid API key

Symptoms:

- `403` when `X-Api-Key` is missing.
- `401 Invalid API key` when key is wrong.
- `500 API key is not configured on the server` when server lacks `API_KEY`.

Resolution:

- Set matching `API_KEY` on server and client/tester.

### LanceDB/Azure storage timeout

Symptoms:

- Rust/Lance warnings about failed range downloads.
- Retrieval timeouts.
- Empty publication results after timeout.

Resolution:

- Retry when storage is stable.
- Increase retrieval/storage timeout environment variables only if this is a persistent latency issue.
- Use `/debug/tables` and `/debug/retrieve` to identify whether tables are reachable.

### Chunk provenance missing

Symptom:

- `validate_corpus_tables.py` reports missing `document_id`/`chunk_id` fields in chunks.

Resolution:

- Run bootstrap metadata first.
- Only run `--rewrite-chunks` when safe for the target LanceDB environment.

### Model endpoint returns only graph or only final ideas

Possible causes:

- Publication retrieval timeout.
- Answer stream timeout.
- Scope guard block.
- Provider error.

Resolution:

- Check server logs for request id.
- Run `scripts/debug_model_stream.py`.
- Run `/debug/retrieve` for the same query.

### Tester returns 404

Possible causes:

- Main API is not running while tester points to local API.
- Tester target base URL is wrong.
- Remote API base URL path is wrong.

Resolution:

- Start the main API with `make run`.
- Start tester with `make run-tester`.
- Confirm tester target is `http://127.0.0.1:8000` for local API.

### Moonshot model endpoints return 503

Possible causes:

- No Azure OpenAI or OpenAI credentials configured.
- Provider authentication or upstream failure.

Resolution:

- Check `/api/moonshot/health`.
- Verify Moonshot model environment variables.

### Moonshot endpoints return 403

Possible cause:

- Browser origin is not in `ALLOWED_ORIGINS`.

Resolution:

- Add the frontend origin to `ALLOWED_ORIGINS` or test from loopback.

## Copyable assistant kits

New publication-backed assistants should be developed as copyable assistant kits rather than full backend forks.

Primary documentation:

```text
docs/ai-api/copyable-assistant-kits.md
```

Validate a kit:

```bash
python scripts/validate_assistant_kit.py --kit assistant_kits/sample
```

Install a kit into this backend without touching LanceDB:

```bash
python scripts/install_assistant_kit.py --kit assistant_kits/sample
```

Install and import the kit corpus into assistant-specific LanceDB tables:

```bash
python scripts/install_assistant_kit.py --kit assistant_kits/sample --import-corpus --include-chunks
```

Non-SEA assistants must use isolated table names:

```text
{assistant_id}_chunks
{assistant_id}_documents
{assistant_id}_sources
```

Production deployment remains the existing Azure Web App deployment from this repository.

### Export a kit to another repo

Create a portable assistant kit folder in another local repo or workspace:

```bash
python scripts/export_assistant_kit.py \
  --kit assistant_kits/_template \
  --target /path/to/other-repo/my_assistant_kit
```

Tell Codex in that repo to read `my_assistant_kit/CODEX_HANDOFF.md` before editing. To bring the finished kit back, run `scripts/install_assistant_kit.py --kit /path/to/other-repo/my_assistant_kit --overwrite` from this backend repo.
