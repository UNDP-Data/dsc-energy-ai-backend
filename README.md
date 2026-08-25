# dsc-sea-ai-api

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License](https://img.shields.io/github/license/undp-data/dsc-sea-ai-api)](https://github.com/undp-data/dsc-sea-ai-api/blob/main/LICENSE)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)
[![Build and deploy Python app to Azure Web App](https://github.com/UNDP-Data/dsc-sea-ai-api/actions/workflows/azure-webapps-python.yml/badge.svg)](https://github.com/UNDP-Data/dsc-sea-ai-api/actions/workflows/azure-webapps-python.yml)

A Python API to serve data from the knowledge graph for the Sustainable Energy Academy.

> [!WARNING]  
> The package is currently undergoing a major revamp. Some features may be missing or not working as intended. Feel free to [open an issue](https://github.com/UNDP-Data/dsc-sea-ai-api/issues).

## Table of Contents

- [Getting Started](#getting-started)
- [API Structure](#api-structure)
- [KG Subgraph Tester](#kg-subgraph-tester)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

Follow the steps below to run the API locally.

1. Clone the repository and navigate to the project folder.
2. Create and activate a virtual environment.
3. Create and populate the `.env` file based on `.env.example`.
4. Run `make install` to install project dependencies.
5. To launch the API, run `make run`. The API will be running at http://127.0.0.1:8000.

```bash
git clone https://github.com/UNDP-Data/dsc-sea-ai-api
cd dsc-sea-ai-api
python -m venv .venv
source .venv/bin/activate
make install
make run
# INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

For Azure Blob auth you can use either:
- `STORAGE_SAS_URL` (full SAS URL), or
- `STORAGE_ACCOUNT_NAME` with `STORAGE_ACCOUNT_KEY` / `STORAGE_SAS_TOKEN`.

For local-only development or assistant-kit testing, set `LANCEDB_URI` to a
filesystem directory. When this is set, the backend uses local LanceDB storage
and does not require Azure Blob credentials:

```bash
export LANCEDB_URI="/path/to/local-lancedb"
```

If that local database contains assistant corpus tables but no `nodes`/`edges`
knowledge-graph tables, the API uses a small built-in fallback graph so `/nodes`,
`/graph`, and `/graph/v2` remain testable offline.

Optional local tester env (separate tester app):
- `KG_TESTER_API_KEY` (or fallback `API_KEY`) for tester proxy authentication to backend.
- `KG_TESTER_LOCAL_API_BASE_URL` for local backend target (default `http://127.0.0.1:8000`).
- `KG_TESTER_REMOTE_API_BASE_URL` to prefill remote backend target (default `https://sea-ai-api.azurewebsites.net`).

## API Structure

All protected endpoints require `X-Api-Key`.

### Corpus Metadata Endpoints

- `GET /documents[?pattern=<text>&limit=<int>]`
  - Searches canonical publication records in the `documents` table when available.
  - Response fields include legacy metadata plus additive fields such as:
    - `document_id`
    - `source`
    - `publisher`
    - `document_type`
    - `publication_date`
    - `series_name`
    - `topics`
    - `geographies`

- `GET /sources`
  - Lists source registry records from the `sources` table when available.

### V1 Graph Endpoint

- `GET /graph?query=<text>[&hops=<int>]`
- Response:
  - `nodes[]` with `name`, `description`, `neighbourhood`, `weight`, `colour`
  - `edges[]` with `subject`, `predicate`, `object`, `description`, `weight`

### V2 Graph Endpoint

- `GET /graph/v2?query=<text>`
- Response:
  - `nodes[]` with `name`, `description`, `tier`, `weight`, `colour`
    - `tier` is one of: `central`, `secondary`, `periphery`
  - `edges[]` with `subject`, `predicate`, `object`, `description`, `weight`
  - `level` is not returned in the API payload.

### Model Endpoint Graph Version

- `POST /model[?graph_version=v1|v2]`
- `graph_version` default is `v2`.
- Request body must include at least one message; empty lists return `400`.
- Response includes `X-Request-Id` header for request correlation.
- The stream is NDJSON; graph and text generation run in parallel.
- A deterministic scope guard runs before generation. Off-topic, prompt-extraction,
  and clearly unsafe requests are blocked before they reach the model or retrieval stack.
- Chunk order is not fixed. The graph chunk usually arrives early, but clients must handle it arriving before, during, or after text deltas.
- To avoid stalled streams under slow storage/model conditions, `/model` applies:
  - `MODEL_GRAPH_TIMEOUT_SECONDS` (graph build timeout),
  - `MODEL_STREAM_IDLE_TIMEOUT_SECONDS` (max idle gap between streamed chunks),
  - `MODEL_TOOLS_PREP_TIMEOUT_SECONDS` (SQL/RAG tool preparation timeout),
  - `MODEL_STREAM_WATCHDOG_SECONDS` (overall no-progress watchdog for stream completion),
  - `RETRIEVE_CHUNKS_TIMEOUT_SECONDS` (RAG chunk retrieval timeout).
- This controls the schema of `graph` returned in streamed `/model` chunks:
  - `v1`: legacy graph schema (`neighbourhood` on nodes, `level` on edges)
  - `v2`: staged graph schema (`tier` on nodes, no edge `level`)
- Publication retrieval is now document-centric when a `documents` table is present:
  - the backend ranks canonical document records first,
  - then loads chunks only from shortlisted approved documents,
  - and streams `documents[]` references separately from answer text.

### Corpus Data Model

The retrieval corpus now supports a document-centric layout:

- `sources`
  - canonical source registry with publisher / authority metadata
- `documents`
  - canonical publication records with approval status, source linkage, document type,
    publication date, topical tags, geography tags, and quality signals
- `chunks`
  - chunk-level text records linked to `document_id`

The API remains backward-compatible with the previous chunk-only flow. If `documents`
or `sources` tables do not exist yet, retrieval falls back to the legacy chunk-first path.

### Internal KG Module Layout

Knowledge graph logic is versioned under:
- `src/kg/v1.py` for V1 graph assembly
- `src/kg/v2.py` for V2 staged graph assembly
- `src/kg/types.py` for V2 response models

Compatibility adapters are retained:
- `src/entities_v2.py`
- `src/graph_v2.py`

## KG Subgraph Tester

The KG tester is now a separate frontend app in `frontend/`. The backend API deployment
does not include the tester route.

1. Start the API backend locally (`make run`).
2. Start the tester frontend app (`make run-tester`).
3. Open [http://127.0.0.1:8010/kg-tester](http://127.0.0.1:8010/kg-tester).
4. Enter:
   - target (`Local server` or `Remote API`),
   - graph version (`default`, `v1`, or `v2`, sent as `graph_version` to `/model` when selected),
   - a graph `query` (for example `climate change mitigation`),
   - remote API base URL (only when target is `Remote API`).
5. Submit to call `/model` and view:
   - streamed answer text (delta chunks),
   - query ideas (clickable chips that trigger a new query),
   - graph payload returned in the stream,
   - an interactive D3 force graph,
   - the raw JSON response payload.

Interaction shortcuts:
- Press `Enter` in the form to run the current query.
- Click a graph node to run a follow-up query (`tell me more about <node>`).
- Click an idea chip to run that suggestion immediately.

Notes:
- The tester proxy uses `KG_TESTER_API_KEY` (or `API_KEY`) from environment and does not expose
  API keys in browser JavaScript.
- The tester proxy forwards `X-Request-Id` (or generates one) to simplify backend trace correlation.
- The tester app calls backend APIs server-to-server, so browser CORS is not required for remote targets.
- No extra CORS proxy is needed when using the standalone tester app.
- The tester app is local-only by design (loopback clients only).

Example local startup:

```bash
# terminal 1
make run

# terminal 2
export KG_TESTER_API_KEY="$API_KEY"
make run-tester
```

`make run-tester` binds to `127.0.0.1:8010` by default.

## SGP AI Tester

The SGP AI tester is a separate local-only frontend for the `sgp_ai` assistant.
It proxies to `/assistants/sgp_ai/model` and `/assistants/sgp_ai/debug/retrieve`
without exposing API keys in browser JavaScript.

Recommended startup with automatic port fallback:

```bash
cd "/Users/ben/Documents/UNDP/SEH/dsc-energy-ai-backend"
.venv/bin/python scripts/run_sgp_ai_local.py --reload
```

The launcher starts the backend and tester UI, reuses an existing backend when
one is already serving the API, and moves the tester to the next free port when
`8015` is occupied. Use the `Tester UI:` URL printed by the launcher.

Example local startup when port `8000` is already in use:

```bash
# terminal 1
.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8016

# terminal 2
SGP_TESTER_API_BASE_URL=http://127.0.0.1:8016 make run-sgp-tester
```

If `8015` is already in use, set a different tester port manually:

```bash
SGP_TESTER_API_BASE_URL=http://127.0.0.1:8016 SGP_TESTER_PORT=8017 make run-sgp-tester
```

Open [http://127.0.0.1:8015/sgp-ai-tester](http://127.0.0.1:8015/sgp-ai-tester),
or the alternate tester URL printed by the launcher/manual port you selected.

The tester checks both assistant profile installation and corpus readiness. If it
shows `SGP AI installed; corpus missing`, refresh storage credentials and import
the corpus:

```bash
export LANCEDB_URI="/Users/ben/Documents/UNDP/SGP/Innovation Library/.local_lancedb"

.venv/bin/python scripts/install_assistant_kit.py \
  --kit "/Users/ben/Documents/UNDP/SGP/SGP AI/sgp_ai" \
  --overwrite \
  --import-corpus \
  --include-chunks
```

Pre-commit validation:

```bash
python3 -m py_compile main.py src/security.py src/database.py src/genai.py src/entities.py src/kg/__init__.py src/kg/v1.py src/kg/v2.py frontend/kg_tester_app.py tests/test_model.py tests/test_genai.py
make test   # requires dev dependencies installed
```

To evaluate V2 output quality over a representative query set:

```bash
python3 scripts/evaluate_graph_v2.py --api-key "$API_KEY"
```

Optional flags:
- `--queries-file path/to/queries.txt` for custom query sets
- `--output /tmp/graph_v2_eval.json` to persist full metrics

## Corpus Bootstrap and Validation

Use the scripts below to upgrade an existing chunk-only corpus into the new
document-centric layout and validate metadata completeness.

Deploy a dataframe artifact directly to LanceDB. This is the reusable script
form of the original `main.ipynb` table-deployment cells:

```bash
python3 scripts/deploy_lancedb_table.py \
  --input data/chunks-vYY-MM-DD.parquet \
  --table chunks \
  --mode overwrite
```

If a prepared dataframe still needs embeddings, pass a text column and optional
embedded artifact output:

```bash
python3 scripts/deploy_lancedb_table.py \
  --input data/chunks-without-vectors.parquet \
  --table chunks \
  --embed-text-column content \
  --embedded-output data/chunks-with-vectors.parquet \
  --mode overwrite
```

By default the backend uses Azure-backed `az://lancedb`. If `LANCEDB_URI` is set,
the same command writes to that local or explicit LanceDB URI instead.
Assistant profiles can also define isolated storage, for example SGP uses
`az://lancedb/sgp_ai` while SEA keeps the default `az://lancedb`. A local
`LANCEDB_URI` override still wins for prototype runs.

Bootstrap canonical `sources` and `documents` tables from the current `chunks` table:

```bash
python3 scripts/bootstrap_corpus_tables.py --overwrite
```

If you also want to rewrite the `chunks` table so every chunk includes `document_id`
and provenance fields:

```bash
python3 scripts/bootstrap_corpus_tables.py --overwrite --rewrite-chunks
```

Notes:
- `--overwrite` alone creates / refreshes `sources` and `documents`, but leaves the
  existing `chunks` schema untouched.
- `--rewrite-chunks` is required if you want chunk provenance fields such as
  `document_id`, `chunk_id`, and `chunk_index` to exist in the `chunks` table.
- Importing enriched manifest chunks also requires `--rewrite-chunks` to have been run.

Validate the upgraded corpus:

```bash
python3 scripts/validate_corpus_tables.py
```

The validator reports:
- table existence and counts,
- missing required document metadata,
- document status distribution,
- sample chunk provenance completeness.

Import additional curated documents from a local YAML manifest:

```bash
python3 scripts/import_corpus_manifest.py --manifest data/corpus/sample_manifest.yaml --include-chunks
```

Use the partition deployer for a complete refresh of the governed intranet-
project corpus. Its default mode is read-only; `--apply` embeds missing chunks,
builds and validates staging tables, preserves the Innovation Library partition,
and atomically swaps each validated table:

```bash
python3 scripts/deploy_corpus_partition.py \
  --manifest /path/to/rag-export/corpus/manifest.yaml

python3 scripts/deploy_corpus_partition.py \
  --manifest /path/to/rag-export/corpus/manifest.yaml \
  --apply \
  --allow-non-public-external-embedding
```

The resumable float32 embedding cache is stored beside the manifest unless
`--embedding-cache` is supplied. Do not use the ordinary manifest upsert when
records removed by a policy or classification refresh must also disappear.
The explicit external-embedding flag is required when the manifest contains
non-public or sensitive-flagged records because chunk text is sent to the
configured embedding provider. Use it only after the data transfer is approved.

The public `/pages/sgp-ai` route remains Innovation Library-only. Server-side
MVP access to `innovation_library`, `projects` (alias `project_database`), or
`all` uses `/internal/sgp-ai/*` and the separate
`SGP_PROJECT_RAG_API_KEY`. `all` is the explicit union of the two approved
source IDs, not an unfiltered database query.

Supporting files:
- `data/corpus/sources.yaml`: seed source registry for trusted and partner corpora
- `data/corpus/sample_manifest.yaml`: example manual import manifest

## Retrieval Debug Surface

Use the retrieval debug endpoint to inspect the document-first ranking path for a single query:

```bash
curl -sS "http://127.0.0.1:8000/debug/retrieve?query=tell%20me%20more%20about%20feed%20in%20tariff&limit=12" \
  -H "X-Api-Key: $API_KEY"
```

The response includes:
- selected documents,
- selected chunks,
- retrieval profile,
- query variants,
- retrieval branches taken,
- final selected path (`document_first`, `chunk_lexical`, `chunk_vector`, or `fallback`).

To run a repeatable multi-prompt diagnostic surface and save both retrieval-debug and
`/model` stream outputs:

```bash
python3 scripts/run_retrieval_surface.py --base-url http://127.0.0.1:8000 --output-prefix tmp/retrieval_surface
```

This writes:
- `tmp/retrieval_surface.json`: machine-readable combined results
- `tmp/retrieval_surface.txt`: human-readable report with retrieval debug and raw NDJSON

For stream-only debugging:

```bash
python3 scripts/debug_model_stream.py --base-url http://127.0.0.1:8000 --output tmp/model_stream_debug.txt
```

## Deployment

The project is hooked up to CI/CD via GitHub Actions.
- Workflow: `.github/workflows/azure-webapps-python.yml`
- Azure Web App name: `sea-ai-api`
- A push to `main` triggers deployment.
- The `frontend/` KG tester app is intentionally separate and is not required for API deployment.

## Contributing

All contributions must follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
The codebase is formatted with `black` and `isort`. Use the provided [Makefile](./Makefile) for these
routine operations.

1. Clone or fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Ensure your code is properly formatted (`make format`)
5. Commit your changes (`git commit -m 'Add some feature'`)
6. Push to the branch (`git push origin feature-branch`)
7. Open a pull request

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](./LICENSE) file.
