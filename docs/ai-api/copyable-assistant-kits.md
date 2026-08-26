# Copyable Assistant Kits

## Purpose

Assistant kits are the copyable unit for building a new publication-backed RAG assistant without forking the whole backend.

A kit contains assistant-owned configuration and corpus metadata. The shared backend still owns routing, authentication, Azure OpenAI clients, LanceDB access, streaming, and deployment.

Production deployment stays in this repository. Once a kit is refined, install it back into this backend and deploy through the existing Azure Web App workflow.


## Cross-repo workflow for Codex

Use this when you want to develop a new assistant in another local repository or project folder and bring it back here later.

### 1. Export the template into the other repo

From this backend repo:

```bash
python scripts/export_assistant_kit.py \
  --kit assistant_kits/_template \
  --target /path/to/other-repo/my_assistant_kit
```

This copies only the assistant kit, not the backend runtime. The copied folder contains its own `CODEX_HANDOFF.md` with instructions for Codex in the other repo.

### 2. Tell Codex in the other repo to use the kit docs

The copied kit contains the detailed instructions in `CODEX_HANDOFF.md`. Use a short prompt like this:

```text
Check the assistant kit in this repo and review the documentation there on how to use it. Follow it to create a local version of the RAG assistant that we can prototype here before deploying it back to the original backend. Preserve the documented data surface and folder structure so it can be copied back cleanly.
```

### 3. Bring the refined kit back here

Preferred: run the deploy-back wrapper from inside the other repo:

```bash
python3 "/Users/ben/Documents/UNDP/SEH/dsc-energy-ai-backend/scripts/deploy_assistant_kit.py" \
  --kit ./my_assistant_kit
```

To also import the corpus into LanceDB:

```bash
python3 "/Users/ben/Documents/UNDP/SEH/dsc-energy-ai-backend/scripts/deploy_assistant_kit.py" \
  --kit ./my_assistant_kit \
  --import-corpus \
  --include-chunks
```

### 4. Deploy

Commit the installed files in this backend repo and merge to `main`. The existing Azure workflow deploys the same API app. The assistant is then available under `/assistants/{assistant_id}/...`.

## Kit structure

```text
assistant_kits/{assistant_id}/
  assistant.yaml
  corpus/manifest.yaml
  eval/questions.yaml
  README.md
  tests/
```

Required files:

- `assistant.yaml`: assistant id, display name, domain description, refusal guidance, prompts, table namespace, source definitions, tag rules, and scope terms.
- `corpus/manifest.yaml`: prepared source, document, and chunk records.

Optional files:

- `eval/questions.yaml`: evaluation questions and expected top documents.
- `tests/`: assistant-specific notes or fixtures.

## Table naming

For non-SEA assistants, table names must use the standard namespace:

```text
{assistant_id}_chunks
{assistant_id}_documents
{assistant_id}_sources
```

The `sea` profile is the only supported exception and keeps the existing production table names:

```text
chunks
documents
sources
```

This prevents a copied assistant from accidentally writing into the SEA production corpus.

## Create a new assistant locally

Copy the template folder:

```bash
cp -R assistant_kits/_template /path/to/my_new_assistant
```

Edit:

- `/path/to/my_new_assistant/assistant.yaml`
- `/path/to/my_new_assistant/corpus/manifest.yaml`
- `/path/to/my_new_assistant/eval/questions.yaml`

Use a lowercase assistant id with only letters, digits, and underscores, for example `water_policy`.

## Validate a kit

From this backend repo:

```bash
python scripts/validate_assistant_kit.py --kit /path/to/my_new_assistant
```

The validator checks:

- safe assistant id;
- required profile fields;
- required prompt fields;
- standard table namespace;
- manifest source/document/chunk structure;
- manifest assistant id compatibility.

## Install a kit into this backend

Install profile and copy the kit into `assistant_kits/{assistant_id}`:

```bash
python scripts/install_assistant_kit.py --kit /path/to/my_new_assistant
```

If the profile or kit already exists and should be replaced:

```bash
python scripts/install_assistant_kit.py --kit /path/to/my_new_assistant --overwrite
```

This writes versionable files only. It does not write to LanceDB unless `--import-corpus` is passed.

## Import corpus into LanceDB

After installing, import the manifest into the assistant-specific LanceDB tables:

```bash
python scripts/install_assistant_kit.py --kit /path/to/my_new_assistant --import-corpus --include-chunks
```

Equivalent direct import:

```bash
python scripts/import_corpus_manifest.py \
  --manifest assistant_kits/{assistant_id}/corpus/manifest.yaml \
  --assistant-id {assistant_id} \
  --include-chunks
```

The direct import path is useful for additive corpus iteration after the profile is already installed. It is an ID-based upsert and does not delete rows that disappeared from a later manifest.

For a governed replacement of the SGP intranet-project source partition, use:

```bash
python scripts/deploy_corpus_partition.py \
  --manifest /path/to/rag-export/corpus/manifest.yaml \
  --apply \
  --allow-non-public-external-embedding
```

This path preserves the Innovation Library partition, embeds missing chunks with a resumable cache, validates per-source counts and vector shape in staging tables, and swaps only after validation succeeds.
The external-embedding flag must be used only after approval to send non-public or sensitive-flagged chunk text to the configured provider.

## Run locally

Start the backend:

```bash
source .venv/bin/activate
make run
```

List installed assistants:

```bash
curl -sS -H "X-Api-Key: $API_KEY" http://127.0.0.1:8000/assistants
```

Debug retrieval:

```bash
curl -sS -H "X-Api-Key: $API_KEY" \
  "http://127.0.0.1:8000/assistants/{assistant_id}/debug/retrieve?query=your%20query"
```

Run model stream debug:

```bash
python scripts/debug_model_stream.py \
  --assistant-id {assistant_id} \
  --base-url http://127.0.0.1:8000 \
  --output tmp/{assistant_id}_model_stream_debug.txt
```

Run retrieval surface:

```bash
python scripts/run_retrieval_surface.py \
  --assistant-id {assistant_id} \
  --base-url http://127.0.0.1:8000 \
  --output-prefix tmp/{assistant_id}_retrieval_surface
```

## Deploy

Deployment remains the existing backend deployment:

1. Install the refined kit into this repo.
2. Commit the installed profile and kit files.
3. Merge to `main`.
4. The existing GitHub Actions workflow deploys the same Azure Web App.

New assistants are served from route-prefixed endpoints:

- `POST /assistants/{assistant_id}/model`
- `GET /assistants/{assistant_id}/documents`
- `GET /assistants/{assistant_id}/sources`
- `GET /assistants/{assistant_id}/debug/retrieve`

SEA legacy endpoints remain unchanged.

## Assistant-specific vs core runtime code

Assistant-specific:

- profile YAML;
- manifest YAML;
- source definitions;
- prompts;
- tag/scope rules;
- evaluation questions.

Core runtime:

- FastAPI routes;
- authentication;
- Azure OpenAI model/embedding clients;
- LanceDB repository access;
- streaming NDJSON contract;
- response safety encoding;
- shared retrieval/ranking code.

Do not copy or modify the core runtime for ordinary new assistants. If a new assistant requires runtime behavior that cannot be expressed in `assistant.yaml`, add it to the shared `src/rag_system` layer and keep it profile-driven where possible.
