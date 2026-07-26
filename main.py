"""
Entry point to the API.
"""

import asyncio
from contextlib import asynccontextmanager, suppress
from inspect import isawaitable
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4
import json
import logging
import os
from time import monotonic

import networkx as nx
import yaml
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from packaging.version import Version

from src import corpus, database, genai
from src.entities import (
    AssistantResponse,
    Document,
    Graph,
    GraphParameters,
    Message,
    Node,
    SourceRecord,
)
from src.kg import v1 as kg_v1
from src.kg import v2 as kg_v2
from src.kg.types import GraphV2, GraphV2Parameters
from src.moonshot import router as moonshot_router
from src.rag_system import RagProfileError, get_profile, list_profiles
from src.security import authenticate

load_dotenv()
logger = logging.getLogger(__name__)
SGP_AI_PAGES_ASSISTANT_ID = "sgp_ai"
SgpAiDataSource = Literal["innovation_library"]
AssistantUiLocale = Literal["en", "pt", "fr", "es", "ru", "zh", "ar"]
SGP_AI_PAGES_SOURCE_IDS = ("gef_sgp_innovation_library",)


def _sgp_ai_pages_source_ids(_data_source: SgpAiDataSource) -> tuple[str, ...]:
    """
    Restrict public SGP AI Pages requests to the approved Innovation Library.
    """
    return SGP_AI_PAGES_SOURCE_IDS


def _env_timeout_seconds(
    name: str,
    default: float,
    *,
    min_value: float = 0.1,
) -> float:
    """
    Parse timeout configuration from environment with safe fallback.

    Invalid, empty, or too-small values fall back to the provided default.
    """
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        logger.warning(
            "Invalid %s value %r; using default=%s",
            name,
            raw,
            default,
        )
        return default
    if parsed < min_value:
        logger.warning(
            "Ignoring %s value %s below minimum %s; using default=%s",
            name,
            parsed,
            min_value,
            default,
        )
        return default
    return parsed


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Define the logic that should be executed before the application start up.

    See the details in [the documentation](https://fastapi.tiangolo.com/advanced/events/).

    Parameters
    ----------
    _ : FastAPI
        Application object, not directly used but required by signature.

    Yields
    ------
    states : dict
        Dictionary of arbitrary state variables.
    """
    connection = await database.get_connection()
    client = database.Client(connection)
    states = {"client": client, "graph": await client.get_knowledge_graph()}
    yield states


with open("metadata.yaml", "r", encoding="utf-8") as file:
    metadata = yaml.safe_load(file)
app = FastAPI(**metadata, lifespan=lifespan)
app.include_router(moonshot_router)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates/")


def _get_profile_or_404(assistant_id: str):
    try:
        return get_profile(assistant_id)
    except RagProfileError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error


def _sgp_ai_pages_allowed_origins() -> set[str]:
    raw = os.getenv("SGP_AI_PAGES_ALLOWED_ORIGINS") or "https://undp-data.github.io"
    return {item.strip().rstrip("/") for item in raw.split(",") if item.strip()}


def _sgp_ai_pages_origin(request: Request) -> str:
    origin = (request.headers.get("Origin") or "").strip().rstrip("/")
    if not origin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Origin header is required for the SGP AI Pages API.",
        )
    if origin not in _sgp_ai_pages_allowed_origins():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Origin is not allowed for the SGP AI Pages API.",
        )
    return origin


def _sgp_ai_pages_cors_headers(origin: str) -> dict[str, str]:
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Headers": "Content-Type, X-Request-Id",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Expose-Headers": "X-Request-Id",
        "Vary": "Origin",
    }


async def _close_connection(connection) -> None:
    close = getattr(connection, "close", None)
    if close is None:
        return
    result = close()
    if isawaitable(result):
        with suppress(Exception):
            await asyncio.wait_for(result, timeout=2.0)


@asynccontextmanager
async def _profile_client(profile):
    connection = await database.get_connection(profile=profile)
    try:
        yield database.Client(connection, profile=profile)
    finally:
        await _close_connection(connection)


@app.get(
    path="/",
    include_in_schema=False,
)
async def root(request: Request):
    """
    Return the homepage template.
    """
    return templates.TemplateResponse(request=request, name="index.html")


@app.get(
    path="/changelog",
    include_in_schema=False,
)
async def changelog(request: Request):
    """
    Return the changelog template.
    """
    file_paths = sorted(
        Path("templates", "changelog").glob("*.html"),
        reverse=True,
        key=lambda path: Version(path.with_suffix("").name.lstrip("v")),
    )
    file_names = [file_path.name for file_path in file_paths]
    return templates.TemplateResponse(
        request=request,
        name="changelog.html",
        context={"file_names": file_names},
    )


@app.get(
    path="/kg-tester",
    include_in_schema=False,
)
async def kg_tester(request: Request):
    """
    Return an interactive page for testing and visualizing `/graph` responses.
    """
    remote_api_base_url = os.getenv("KG_TESTER_REMOTE_API_BASE_URL", "").strip().rstrip("/")
    return templates.TemplateResponse(
        request=request,
        name="kg_tester.html",
        context={"remote_api_base_url": remote_api_base_url},
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    Use to display the favicon in the homepage. For favicon on the documentation pages,
    an override of /docs and /redoc endpoints is required.
    """
    return FileResponse("./static/favicon.ico")


@app.get(
    path="/nodes",
    response_model=list[Node],
    response_model_by_alias=False,
    dependencies=[Depends(authenticate)],
)
async def search_nodes(
    request: Request,
    pattern: Annotated[str | None, Query(max_length=50)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
):
    """
    Search nodes in the graph. Since there is no central node,
    `neighbourhood` is set to zero for all nodes.
    """
    graph: nx.Graph = request.state.graph
    token = (pattern or "").strip().lower()
    nodes: list[Node] = []
    for name, attrs in graph.nodes(data=True):
        if not isinstance(name, str):
            continue
        if token and token not in name.lower():
            continue
        nodes.append(
            Node(
                name=name,
                description=attrs.get("description"),
                weight=float(attrs.get("weight", 0.0) or 0.0),
            )
        )
    return sorted(nodes)[:limit]


@app.get(
    path="/nodes/{name}",
    response_model=Node,
    response_model_by_alias=False,
    dependencies=[Depends(authenticate)],
)
async def get_node(request: Request, name: str):
    """
    Get a single node by name. Case insensitive.
    """
    graph: nx.Graph = request.state.graph
    query = name.lower()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Node '{name}' does not exist.",
        )
    for node_name, attrs in graph.nodes(data=True):
        if isinstance(node_name, str) and node_name.lower() == query:
            return Node(
                name=node_name,
                description=attrs.get("description"),
                weight=float(attrs.get("weight", 0.0) or 0.0),
            )
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Node '{name}' does not exist.",
    )


@app.get(
    path="/documents",
    response_model=list[Document],
    response_model_by_alias=False,
    dependencies=[Depends(authenticate)],
)
async def search_documents(
    request: Request,
    pattern: Annotated[str | None, Query(max_length=120)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    """
    Search canonical publication records if a documents table is available.
    """
    client: database.Client = request.state.client
    table = await client.open_optional_table("documents")
    if table is None:
        return []
    clause = "(status = 'approved')"
    if pattern:
        escaped = pattern.replace("'", "''")
        clause += (
            " AND ("
            f"regexp_like(canonical_title, '(?i){escaped}') OR "
            f"regexp_like(summary, '(?i){escaped}') OR "
            f"regexp_like(topic_tags_text, '(?i){escaped}') OR "
            f"regexp_like(geography_tags_text, '(?i){escaped}')"
            ")"
        )
    rows = await table.query().where(clause).limit(limit).to_list()
    return [corpus.document_record_to_api(row) for row in rows]


@app.get(
    path="/sources",
    response_model=list[SourceRecord],
    response_model_by_alias=False,
    dependencies=[Depends(authenticate)],
)
async def list_sources(request: Request):
    """
    List registered source records if a sources table is available.
    """
    client: database.Client = request.state.client
    table = await client.open_optional_table("sources")
    if table is None:
        return []
    rows = await table.query().limit(200).to_list()
    return [SourceRecord(**row) for row in rows]


@app.get(
    path="/assistants",
    dependencies=[Depends(authenticate)],
)
async def list_rag_assistants():
    """
    List configured publication-backed RAG assistant profiles.
    """
    return [
        {
            "assistant_id": profile.assistant_id,
            "display_name": profile.display_name,
            "domain_description": profile.domain_description,
            "tables": profile.table_names,
            "default": profile.is_default,
        }
        for profile in list_profiles()
    ]


@app.get(
    path="/assistants/{assistant_id}/documents",
    response_model=list[Document],
    response_model_by_alias=False,
    dependencies=[Depends(authenticate)],
)
async def search_assistant_documents(
    assistant_id: str,
    pattern: Annotated[str | None, Query(max_length=120)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    """
    Search canonical publication records for a configured RAG assistant.
    """
    profile = _get_profile_or_404(assistant_id)
    async with _profile_client(profile) as client:
        table = await client.open_optional_table(client.table_name("documents"))
        if table is None:
            return []
        clause = "(status = 'approved')"
        if pattern:
            escaped = pattern.replace("'", "''")
            clause += (
                " AND ("
                f"regexp_like(canonical_title, '(?i){escaped}') OR "
                f"regexp_like(summary, '(?i){escaped}') OR "
                f"regexp_like(topic_tags_text, '(?i){escaped}') OR "
                f"regexp_like(geography_tags_text, '(?i){escaped}')"
                ")"
            )
        rows = await table.query().where(clause).limit(limit).to_list()
        return [corpus.document_record_to_api(row) for row in rows]


@app.get(
    path="/assistants/{assistant_id}/sources",
    response_model=list[SourceRecord],
    response_model_by_alias=False,
    dependencies=[Depends(authenticate)],
)
async def list_assistant_sources(assistant_id: str):
    """
    List source records for a configured RAG assistant.
    """
    profile = _get_profile_or_404(assistant_id)
    async with _profile_client(profile) as client:
        table = await client.open_optional_table(client.table_name("sources"))
        if table is None:
            return []
        rows = await table.query().limit(200).to_list()
        return [SourceRecord(**row) for row in rows]


@app.get(
    path="/assistants/{assistant_id}/debug/retrieve",
    dependencies=[Depends(authenticate)],
    include_in_schema=False,
)
async def debug_assistant_retrieve(
    assistant_id: str,
    query: Annotated[str, Query(min_length=2)],
    limit: Annotated[int, Query(ge=1, le=50)] = 12,
):
    """
    Inspect document/chunk retrieval decisions for a configured assistant.
    """
    profile = _get_profile_or_404(assistant_id)
    async with _profile_client(profile) as client:
        debug_payload: dict = {}
        chunks, documents = await client.retrieve_chunks(query, limit=limit, debug=debug_payload)
        return {
            "assistant_id": profile.assistant_id,
            "query": query,
            "limit": limit,
            "documents": [document.model_dump() for document in documents],
            "chunks": [chunk.model_dump() for chunk in chunks],
            "debug": debug_payload,
        }


@app.options(
    path="/pages/sgp-ai/{path:path}",
    include_in_schema=False,
)
async def sgp_ai_pages_preflight(request: Request):
    """
    CORS preflight for the static GitHub Pages SGP AI interface.
    """
    origin = _sgp_ai_pages_origin(request)
    return Response(status_code=204, headers=_sgp_ai_pages_cors_headers(origin))


@app.get(
    path="/pages/sgp-ai/status",
    include_in_schema=False,
)
async def sgp_ai_pages_status(
    request: Request,
    data_source: Annotated[SgpAiDataSource, Query()] = "innovation_library",
):
    """
    Origin-limited readiness check for the static GitHub Pages SGP AI interface.
    """
    origin = _sgp_ai_pages_origin(request)
    source_ids = _sgp_ai_pages_source_ids(data_source)
    profile = _get_profile_or_404(SGP_AI_PAGES_ASSISTANT_ID)
    async with _profile_client(profile) as client:
        table = await client.open_optional_table(client.table_name("documents"))
        document_count = 0
        if table is not None:
            source_filter = None
            if source_ids:
                source_filter = " OR ".join(
                    f"source_id = '{source_id}'" for source_id in source_ids
                )
            document_count = await table.count_rows(source_filter)
    return JSONResponse(
        {
            "ok": True,
            "assistant_id": profile.assistant_id,
            "display_name": profile.display_name,
            "corpus_ready": document_count > 0,
            "document_count": document_count,
            "data_source": data_source,
        },
        headers=_sgp_ai_pages_cors_headers(origin),
    )


@app.get(
    path="/pages/sgp-ai/debug/retrieve",
    include_in_schema=False,
)
async def sgp_ai_pages_retrieve(
    request: Request,
    query: Annotated[str, Query(min_length=2)],
    limit: Annotated[int, Query(ge=1, le=12)] = 6,
    data_source: Annotated[SgpAiDataSource, Query()] = "innovation_library",
):
    """
    Origin-limited retrieval preview for the static GitHub Pages SGP AI interface.
    """
    origin = _sgp_ai_pages_origin(request)
    profile = _get_profile_or_404(SGP_AI_PAGES_ASSISTANT_ID)
    source_ids = _sgp_ai_pages_source_ids(data_source)
    async with _profile_client(profile) as client:
        chunks, documents = await client.retrieve_chunks(
            query,
            limit=limit,
            source_ids=source_ids,
        )
    return JSONResponse(
        {
            "assistant_id": profile.assistant_id,
            "query": query,
            "limit": limit,
            "data_source": data_source,
            "documents": [document.model_dump() for document in documents],
            "chunks": [chunk.model_dump() for chunk in chunks],
        },
        headers=_sgp_ai_pages_cors_headers(origin),
    )


@app.get(
    path="/pages/sgp-ai/relevance-map",
    include_in_schema=False,
)
async def sgp_ai_pages_relevance_map(
    request: Request,
    query: Annotated[str, Query(min_length=2)],
    data_source: Annotated[SgpAiDataSource, Query()] = "innovation_library",
):
    """
    Origin-limited document-level relevance scores for the static SGP AI interface.
    """
    origin = _sgp_ai_pages_origin(request)
    profile = _get_profile_or_404(SGP_AI_PAGES_ASSISTANT_ID)
    source_ids = _sgp_ai_pages_source_ids(data_source)
    async with _profile_client(profile) as client:
        documents = await client.score_document_relevance_map(
            query,
            source_ids=source_ids,
        )
    return JSONResponse(
        {
            "assistant_id": profile.assistant_id,
            "query": query,
            "data_source": data_source,
            "document_count": len(documents),
            "documents": documents,
        },
        headers=_sgp_ai_pages_cors_headers(origin),
    )


@app.post(
    path="/pages/sgp-ai/model",
    response_model=AssistantResponse,
    response_model_by_alias=False,
    include_in_schema=False,
)
async def sgp_ai_pages_model(
    request: Request,
    messages: list[Message],
    data_source: Annotated[SgpAiDataSource, Query()] = "innovation_library",
    ui_locale: Annotated[AssistantUiLocale, Query()] = "en",
):
    """
    Origin-limited streaming answer endpoint for the static GitHub Pages SGP AI interface.
    """
    origin = _sgp_ai_pages_origin(request)
    request.state.retrieval_source_ids = _sgp_ai_pages_source_ids(data_source)
    request.state.retrieval_data_source = data_source
    request.state.ui_locale = ui_locale
    response = await ask_assistant_model(request, SGP_AI_PAGES_ASSISTANT_ID, messages)
    for key, value in _sgp_ai_pages_cors_headers(origin).items():
        response.headers[key] = value
    return response


@app.get(
    path="/graph",
    response_model=Graph,
    response_model_by_alias=False,
    dependencies=[Depends(authenticate)],
)
async def query_knowledge_graph(
    request: Request,
    params: Annotated[GraphParameters, Query()],
):
    """
    Get a knowledge graph that best matches the query concept.
    """
    client: database.Client = request.state.client
    graph: nx.Graph = request.state.graph
    return await kg_v1.build_subgraph_v1(client, graph, **params.model_dump())


@app.get(
    path="/graph/v2",
    response_model=GraphV2,
    response_model_by_alias=False,
    dependencies=[Depends(authenticate)],
)
async def query_knowledge_graph_v2(
    request: Request,
    response: Response,
    params: Annotated[GraphV2Parameters, Query()],
):
    """
    Get a staged knowledge graph with central / secondary / periphery tiers.
    """
    client: database.Client = request.state.client
    graph: nx.Graph = request.state.graph
    timings: dict[str, float] = {}
    result = await kg_v2.build_subgraph_v2(client, graph, params.query, timings=timings)
    response.headers["X-KG-Timing"] = json.dumps(timings, separators=(",", ":"))
    server_timing = []
    key_to_metric = {
        "central_exact_match_ms": "central_exact",
        "central_lexical_ms": "central_lexical",
        "central_embed_ms": "central_embed",
        "central_vector_search_ms": "central_search",
        "central_selection_ms": "central",
        "stage1_ms": "stage1",
        "stage2_ms": "stage2",
        "stage3_ms": "stage3",
        "total_ms": "total",
    }
    for key, metric in key_to_metric.items():
        if key in timings:
            server_timing.append(f"{metric};dur={timings[key]:.2f}")
    if server_timing:
        response.headers["Server-Timing"] = ", ".join(server_timing)
    return result


@app.get(
    path="/debug/tables",
    dependencies=[Depends(authenticate)],
    include_in_schema=False,
)
async def debug_tables(request: Request):
    """
    Return table presence and row counts to validate storage wiring.
    """
    async def maybe_await(value):
        return await value if isawaitable(value) else value

    client: database.Client = request.state.client
    names = await maybe_await(client.connection.table_names())
    names = list(names)
    status = {}
    for name in ("nodes", "edges", "chunks", "documents", "sources", "sdg7"):
        if name in names:
            try:
                table = await maybe_await(client.connection.open_table(name))
                rows = await maybe_await(table.count_rows())
                status[name] = {"exists": True, "rows": rows}
            except Exception as error:
                status[name] = {"exists": True, "rows": None, "error": str(error)}
        else:
            status[name] = {"exists": False, "rows": 0}
    return {"tables": status, "all_tables": sorted(names)}


@app.get(
    path="/debug/retrieve",
    dependencies=[Depends(authenticate)],
    include_in_schema=False,
)
async def debug_retrieve(
    request: Request,
    query: Annotated[str, Query(min_length=2)],
    limit: Annotated[int, Query(ge=1, le=50)] = 12,
):
    """
    Inspect document/chunk retrieval decisions for a single query.
    """
    client: database.Client = request.state.client
    debug_payload: dict = {}
    chunks, documents = await client.retrieve_chunks(query, limit=limit, debug=debug_payload)
    return {
        "query": query,
        "limit": limit,
        "documents": [document.model_dump() for document in documents],
        "chunks": [chunk.model_dump() for chunk in chunks],
        "debug": debug_payload,
    }


@app.post(
    path="/assistants/{assistant_id}/model",
    response_model=AssistantResponse,
    response_model_by_alias=False,
    dependencies=[Depends(authenticate)],
)
async def ask_assistant_model(
    request: Request,
    assistant_id: str,
    messages: list[Message],
    ui_locale: Annotated[AssistantUiLocale, Query()] = "en",
):
    """
    Ask a configured RAG assistant to stream an NDJSON response.

    This route is RAG-only in v1 and intentionally does not emit Knowledge Graph
    chunks for non-SEA assistants.
    """
    if not messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one message is required.",
        )
    if messages[-1].role != "human":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The last message must come from the user.",
        )

    profile = _get_profile_or_404(assistant_id)
    request_id = request.headers.get("X-Request-Id") or uuid4().hex
    user_query = messages[-1].content
    effective_ui_locale = genai.normalize_ui_locale(
        getattr(request.state, "ui_locale", ui_locale)
    )
    fallback_answer_locale = genai.infer_question_locale(
        user_query,
        effective_ui_locale,
    )
    retrieval_source_ids = getattr(request.state, "retrieval_source_ids", None)
    scope_decision = genai.assess_profile_scope(messages, profile)
    if not scope_decision.allowed:
        logger.warning(
            "Blocked assistant request assistant_id=%s request_id=%s category=%s reason=%s query=%r",
            profile.assistant_id,
            request_id,
            scope_decision.category,
            scope_decision.reason,
            user_query,
        )

        async def stream_blocked_response():
            if profile.assistant_id == SGP_AI_PAGES_ASSISTANT_ID:
                refusal = genai.localized_assistant_text(
                    "outside_sgp_scope",
                    fallback_answer_locale,
                )
                ideas = genai.localized_scope_ideas(effective_ui_locale)
            else:
                refusal = scope_decision.refusal or "This request is outside the supported scope."
                ideas = genai.build_scope_ideas(scope_decision.category)
            yield (
                AssistantResponse(
                    role="assistant",
                    content=refusal,
                    graph=None,
                ).model_dump_json()
                + "\n"
            )
            yield (
                AssistantResponse(
                    role="assistant",
                    content="",
                    ideas=ideas,
                    graph=None,
                ).model_dump_json()
                + "\n"
            )

        return StreamingResponse(
            content=stream_blocked_response(),
            media_type="application/x-ndjson",
            headers={"X-Request-Id": request_id},
        )

    logger.info(
        "Assistant model request started assistant_id=%s request_id=%s",
        profile.assistant_id,
        request_id,
    )

    async def maybe_close(connection) -> None:
        close = getattr(connection, "close", None)
        if close is None:
            return
        result = close()
        if isawaitable(result):
            with suppress(Exception):
                await asyncio.wait_for(result, timeout=2.0)

    async def publication_payload():
        retrieval_timeout_seconds = _env_timeout_seconds(
            "MODEL_PUBLICATION_RETRIEVAL_TIMEOUT_SECONDS",
            90.0,
        )
        answer_connection = None
        translation_task: asyncio.Task[str] | None = None
        if (
            profile.assistant_id == SGP_AI_PAGES_ASSISTANT_ID
            and fallback_answer_locale != "en"
        ):
            translation_task = asyncio.create_task(
                genai.translate_query_for_retrieval(user_query)
            )
        logger.info(
            "Assistant publication retrieval scheduled assistant_id=%s request_id=%s query=%r timeout=%ss",
            profile.assistant_id,
            request_id,
            user_query,
            retrieval_timeout_seconds,
        )
        try:
            answer_connection = await database.get_connection(profile=profile)
            answer_client = database.Client(answer_connection, profile=profile)
            retrieval_query = user_query
            query_variants: list[str] = []
            if translation_task is not None:
                translation_timeout_seconds = _env_timeout_seconds(
                    "MODEL_RETRIEVAL_TRANSLATION_TIMEOUT_SECONDS",
                    10.0,
                )
                try:
                    translated_query = await asyncio.wait_for(
                        translation_task,
                        timeout=translation_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Assistant retrieval translation timed out assistant_id=%s request_id=%s timeout=%ss",
                        profile.assistant_id,
                        request_id,
                        translation_timeout_seconds,
                    )
                except Exception as error:
                    logger.warning(
                        "Assistant retrieval translation failed assistant_id=%s request_id=%s: %s",
                        profile.assistant_id,
                        request_id,
                        error,
                    )
                else:
                    normalized_original = " ".join(user_query.lower().split())
                    normalized_translation = " ".join(translated_query.lower().split())
                    if translated_query and normalized_translation != normalized_original:
                        retrieval_query = translated_query
                        query_variants = [user_query]
            started_at = monotonic()
            try:
                retrieval_kwargs = {}
                if retrieval_source_ids:
                    retrieval_kwargs["source_ids"] = retrieval_source_ids
                if query_variants:
                    retrieval_kwargs["query_variants"] = query_variants
                chunks, documents = await asyncio.wait_for(
                    answer_client.retrieve_chunks(retrieval_query, **retrieval_kwargs),
                    timeout=retrieval_timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Assistant publication retrieval timed out assistant_id=%s request_id=%s query=%r timeout=%ss",
                    profile.assistant_id,
                    request_id,
                    user_query,
                    retrieval_timeout_seconds,
                )
                return [], []
            except Exception as error:
                logger.exception(
                    "Assistant publication retrieval failed assistant_id=%s request_id=%s query=%r: %s",
                    profile.assistant_id,
                    request_id,
                    user_query,
                    error,
                )
                return [], []
            logger.info(
                "Assistant publication retrieval completed assistant_id=%s request_id=%s query=%r chunks=%s documents=%s elapsed_ms=%.2f",
                profile.assistant_id,
                request_id,
                user_query,
                len(chunks),
                len(documents),
                (monotonic() - started_at) * 1000,
            )
            return [chunk.to_context() for chunk in chunks], documents
        finally:
            if translation_task is not None and not translation_task.done():
                translation_task.cancel()
                with suppress(asyncio.CancelledError):
                    await translation_task
            if answer_connection is not None:
                await maybe_close(answer_connection)

    async def stream_assistant_response():
        response = AssistantResponse(role="assistant", content="", graph=None)
        stream_idle_timeout_seconds = _env_timeout_seconds(
            "MODEL_STREAM_IDLE_TIMEOUT_SECONDS",
            90.0,
        )
        defer_initial_answer = (
            not profile.is_default
            or database.should_defer_to_publications(user_query)
        )
        answer_iter = genai.get_answer(
            messages,
            response,
            publication_task=publication_payload(),
            defer_initial_answer=defer_initial_answer,
            profile=profile,
            ui_locale=effective_ui_locale,
        )
        emitted_chunks = 0
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        answer_iter.__anext__(),
                        timeout=stream_idle_timeout_seconds,
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    logger.warning(
                        "Assistant model stream idle timeout assistant_id=%s request_id=%s timeout=%ss",
                        profile.assistant_id,
                        request_id,
                        stream_idle_timeout_seconds,
                    )
                    fallback = AssistantResponse(
                        role="assistant",
                        content=genai.localized_assistant_text(
                            "temporary_answer_issue",
                            fallback_answer_locale,
                        ),
                        graph=None,
                    )
                    yield fallback.model_dump_json() + "\n"
                    emitted_chunks += 1
                    break
                if await request.is_disconnected():
                    logger.info(
                        "Client disconnected during assistant stream assistant_id=%s request_id=%s.",
                        profile.assistant_id,
                        request_id,
                    )
                    return
                yield chunk
                emitted_chunks += 1
        finally:
            with suppress(Exception):
                await asyncio.wait_for(answer_iter.aclose(), timeout=1.0)
        logger.info(
            "Assistant model request finished assistant_id=%s request_id=%s chunks=%s",
            profile.assistant_id,
            request_id,
            emitted_chunks,
        )

    return StreamingResponse(
        content=stream_assistant_response(),
        media_type="application/x-ndjson",
        headers={"X-Request-Id": request_id},
    )


@app.post(
    path="/model",
    response_model=AssistantResponse,
    response_model_by_alias=False,
    dependencies=[Depends(authenticate)],
)
async def ask_model(
    request: Request,
    messages: list[Message],
    graph_version: Annotated[
        Literal["v1", "v2"],
        Query(
            description="Knowledge graph response schema version.",
            json_schema_extra={"example": "v2"},
        ),
    ] = "v2",
):
    """
    Ask a GenAI model to compose a response supplemented by knowledge graph data.

    Messages are expected to be in chronological order, i.e., from the oldest to most recent,
    with the current user message being the last one. This endpoint streams the response in
    NDJSON format.
    """
    if not messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one message is required.",
        )
    if messages[-1].role != "human":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The last message must come from the user.",
        )
    request_id = request.headers.get("X-Request-Id") or uuid4().hex
    user_query = messages[-1].content
    graph: nx.Graph = request.state.graph
    scope_decision = genai.assess_scope(messages)
    if not scope_decision.allowed:
        logger.warning(
            "Blocked model request request_id=%s category=%s reason=%s query=%r",
            request_id,
            scope_decision.category,
            scope_decision.reason,
            user_query,
        )

        async def stream_blocked_response():
            empty_graph: Graph | GraphV2
            if graph_version == "v1":
                empty_graph = Graph(nodes=[], edges=[])
            else:
                empty_graph = GraphV2(nodes=[], edges=[])
            yield (
                AssistantResponse(
                    role="assistant",
                    content="",
                    graph=empty_graph,
                ).model_dump_json()
                + "\n"
            )
            yield (
                AssistantResponse(
                    role="assistant",
                    content=scope_decision.refusal or "This request is outside the supported scope.",
                    graph=None,
                ).model_dump_json()
                + "\n"
            )
            yield (
                AssistantResponse(
                    role="assistant",
                    content="",
                    ideas=genai.build_scope_ideas(scope_decision.category),
                    graph=None,
                ).model_dump_json()
                + "\n"
            )

        return StreamingResponse(
            content=stream_blocked_response(),
            media_type="application/x-ndjson",
            headers={"X-Request-Id": request_id},
        )

    logger.info(
        "Model request started request_id=%s graph_version=%s",
        request_id,
        graph_version,
    )

    async def stream_model_response():
        emitted_chunks = 0
        if await request.is_disconnected():
            logger.info(
                "Client disconnected before /model stream started request_id=%s.",
                request_id,
            )
            return
        queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue()
        graph_timeout_seconds = _env_timeout_seconds(
            "MODEL_GRAPH_TIMEOUT_SECONDS", 120.0
        )
        stream_idle_timeout_seconds = _env_timeout_seconds(
            "MODEL_STREAM_IDLE_TIMEOUT_SECONDS", 90.0
        )
        tools_prep_timeout_seconds = _env_timeout_seconds(
            "MODEL_TOOLS_PREP_TIMEOUT_SECONDS", 45.0
        )
        stream_watchdog_seconds = _env_timeout_seconds(
            "MODEL_STREAM_WATCHDOG_SECONDS", 150.0
        )

        async def maybe_close(connection) -> None:
            close = getattr(connection, "close", None)
            if close is None:
                return
            result = close()
            if isawaitable(result):
                with suppress(Exception):
                    await asyncio.wait_for(result, timeout=2.0)

        async def produce_graph() -> None:
            fallback_graph: Graph | GraphV2
            if graph_version == "v1":
                fallback_graph = Graph(nodes=[], edges=[])
            else:
                fallback_graph = GraphV2(nodes=[], edges=[])
            graph_connection = None
            try:
                # Use a dedicated connection to avoid lock/contention with concurrent answer flow.
                graph_connection = await database.get_connection()
                graph_client = database.Client(graph_connection)
                if graph_version == "v1":
                    entities = await genai.extract_entities(user_query)
                    response_graph = await asyncio.wait_for(
                        graph_client.find_subgraph(graph, entities),
                        timeout=graph_timeout_seconds,
                    )
                else:
                    response_graph = await asyncio.wait_for(
                        kg_v2.build_subgraph_v2(graph_client, graph, user_query),
                        timeout=graph_timeout_seconds,
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    "Graph build timed out request_id=%s timeout=%ss",
                    request_id,
                    graph_timeout_seconds,
                )
                response_graph = fallback_graph
            except Exception as error:
                logger.exception(
                    "Failed to build response graph request_id=%s: %s",
                    request_id,
                    error,
                )
                response_graph = fallback_graph
            finally:
                if graph_connection is not None:
                    await maybe_close(graph_connection)
            if await request.is_disconnected():
                logger.info(
                    "Client disconnected before graph chunk was sent request_id=%s.",
                    request_id,
                )
                return
            graph_response = AssistantResponse(
                role="assistant",
                content="",
                graph=response_graph,
            )
            await queue.put(("graph_data", graph_response.model_dump_json() + "\n"))

        async def produce_answer() -> None:
            response = AssistantResponse(
                role="assistant",
                content="",
                graph=None,
            )
            async def publication_payload():
                retrieval_timeout_seconds = _env_timeout_seconds(
                    "MODEL_PUBLICATION_RETRIEVAL_TIMEOUT_SECONDS",
                    90.0,
                )
                answer_connection = None
                logger.info(
                    "Publication retrieval scheduled request_id=%s query=%r timeout=%ss",
                    request_id,
                    user_query,
                    retrieval_timeout_seconds,
                )
                try:
                    answer_connection = await database.get_connection()
                    answer_client = database.Client(answer_connection)
                    started_at = monotonic()
                    logger.info(
                        "Publication retrieval started request_id=%s query=%r timeout=%ss",
                        request_id,
                        user_query,
                        retrieval_timeout_seconds,
                    )
                    try:
                        chunks, documents = await asyncio.wait_for(
                            answer_client.retrieve_chunks(user_query),
                            timeout=retrieval_timeout_seconds,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Publication retrieval timed out request_id=%s query=%r timeout=%ss",
                            request_id,
                            user_query,
                            retrieval_timeout_seconds,
                        )
                        return [], []
                    except Exception as error:
                        logger.exception(
                            "Publication retrieval failed request_id=%s query=%r: %s",
                            request_id,
                            user_query,
                            error,
                        )
                        return [], []
                    logger.info(
                        "Publication retrieval completed request_id=%s query=%r chunks=%s documents=%s elapsed_ms=%.2f",
                        request_id,
                        user_query,
                        len(chunks),
                        len(documents),
                        (monotonic() - started_at) * 1000,
                    )
                    if chunks or documents:
                        logger.info(
                            "Publication payload ready request_id=%s query=%r chunks=%s documents=%s",
                            request_id,
                            user_query,
                            len(chunks),
                            len(documents),
                        )
                        return [chunk.to_context() for chunk in chunks], documents
                finally:
                    if answer_connection is not None:
                        await maybe_close(answer_connection)
                logger.warning(
                    "Publication retrieval returned no usable results request_id=%s query=%r",
                    request_id,
                    user_query,
                )
                return [], []

            answer_iter = genai.get_answer(
                messages,
                response,
                publication_task=publication_payload(),
                defer_initial_answer=database.should_defer_to_publications(user_query),
            )
            try:
                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            answer_iter.__anext__(),
                            timeout=stream_idle_timeout_seconds,
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Model stream idle timeout request_id=%s timeout=%ss",
                            request_id,
                            stream_idle_timeout_seconds,
                        )
                        fallback = AssistantResponse(
                            role="assistant",
                            content="I ran into a temporary delay while retrieving supporting data. Please retry your question.",
                            graph=None,
                        )
                        await queue.put(("data", fallback.model_dump_json() + "\n"))
                        break
                    if await request.is_disconnected():
                        logger.info(
                            "Client disconnected during /model token stream request_id=%s.",
                            request_id,
                        )
                        return
                    await queue.put(("answer_data", chunk))
            except Exception as error:
                logger.exception(
                    "Failed while streaming answer request_id=%s: %s",
                    request_id,
                    error,
                )
                fallback = AssistantResponse(
                    role="assistant",
                    content="I ran into a temporary issue while generating a response. Please retry your question.",
                    graph=None,
                )
                await queue.put(("answer_data", fallback.model_dump_json() + "\n"))
            finally:
                with suppress(Exception):
                    await asyncio.wait_for(answer_iter.aclose(), timeout=1.0)

        async def run_producer(done_label: str, producer) -> None:
            try:
                await producer()
            finally:
                await queue.put((done_label, None))

        producer_tasks = [
            asyncio.create_task(run_producer("graph_done", produce_graph)),
            asyncio.create_task(run_producer("answer_done", produce_answer)),
        ]
        completed: set[str] = set()
        graph_emitted = False
        last_event_at = monotonic()
        try:
            while len(completed) < 2:
                if await request.is_disconnected():
                    logger.info(
                        "Client disconnected before /model stream completion request_id=%s.",
                        request_id,
                    )
                    break
                try:
                    label, payload = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if monotonic() - last_event_at >= stream_watchdog_seconds:
                        logger.warning(
                            "Model stream watchdog timeout request_id=%s timeout=%ss",
                            request_id,
                            stream_watchdog_seconds,
                        )
                        if not graph_emitted:
                            empty_graph: Graph | GraphV2
                            if graph_version == "v1":
                                empty_graph = Graph(nodes=[], edges=[])
                            else:
                                empty_graph = GraphV2(nodes=[], edges=[])
                            fallback_graph = AssistantResponse(
                                role="assistant",
                                content="",
                                graph=empty_graph,
                            )
                            yield fallback_graph.model_dump_json() + "\n"
                            emitted_chunks += 1
                            graph_emitted = True
                        fallback = AssistantResponse(
                            role="assistant",
                            content="I ran into a temporary delay while streaming this response. Please retry your question.",
                            graph=None,
                        )
                        yield fallback.model_dump_json() + "\n"
                        emitted_chunks += 1
                        break
                    continue
                last_event_at = monotonic()
                if payload is None:
                    completed.add(label)
                    continue
                if label == "graph_data":
                    graph_emitted = True
                yield payload
                emitted_chunks += 1
        finally:
            for task in producer_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*producer_tasks, return_exceptions=True)
        logger.info(
            "Model request finished request_id=%s chunks=%s",
            request_id,
            emitted_chunks,
        )

    return StreamingResponse(
        content=stream_model_response(),
        media_type="application/x-ndjson",
        headers={"X-Request-Id": request_id},
    )
