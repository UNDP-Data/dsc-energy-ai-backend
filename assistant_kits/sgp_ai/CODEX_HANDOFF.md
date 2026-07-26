# SGP AI Backend Handoff

This directory is the backend-installed copy of the `sgp_ai` assistant kit, not
the working prototype documentation source.

When working in this backend repo, edit this kit only for backend-relevant
changes:

- `assistant.yaml` changes that affect assistant identity, prompts, scope, or
  table names.
- `corpus/manifest.yaml` changes that are ready to be imported into backend
  LanceDB tables.
- `eval/questions.yaml` changes that should be part of backend validation.
- Backend route, proxy, CORS, auth, storage, retrieval, or streaming changes
  outside this kit.

Do not mirror small UI copy edits, GitHub Pages publishing notes, local
prototype instructions, or PDF/chunking process notes into this backend copy.
Those belong in the standalone SGP AI prototype/site repository until a
backend PR is needed.

Preserve the backend contract:

- Assistant ID: `sgp_ai`
- Route prefix: `/assistants/sgp_ai/...`
- Model locale query: `ui_locale=en|pt|fr|es|ru|zh|ar`; answers follow
  the latest question's language, while suggested questions and ambiguous
  questions use the requested interface locale.
- Tables: `sgp_ai_sources`, `sgp_ai_documents`, `sgp_ai_chunks`
