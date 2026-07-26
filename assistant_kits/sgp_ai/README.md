# SGP AI Backend Kit

This is the installed backend copy of the `sgp_ai` assistant kit.

Keep this file limited to backend-facing facts that should change only with a
backend pull request.

- Assistant ID: `sgp_ai`
- Display name: `SGP AI`
- Backend route prefix: `/assistants/sgp_ai/...`
- Model locale query: `ui_locale=en|pt|fr|es|ru|zh|ar`
  - Answers follow the language of the latest user question.
  - Suggested questions use `ui_locale`.
  - Ambiguous or language-neutral questions fall back to `ui_locale`.
- LanceDB tables:
  - `sgp_ai_sources`
  - `sgp_ai_documents`
  - `sgp_ai_chunks`

The backend owns FastAPI routes, API-key authentication, Azure OpenAI clients,
embedding generation, LanceDB access, retrieval, streaming responses, and Azure
deployment.

Do not update this backend copy for small UI text changes, GitHub Pages
publishing notes, local prototyping instructions, or corpus-processing notes.
Keep that documentation in the standalone SGP AI prototype/site repository and
install back into this backend only when there is a backend-relevant assistant
configuration, corpus, evaluation, or runtime change.
