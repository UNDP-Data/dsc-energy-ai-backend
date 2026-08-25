"""
Entities (models) and related routines to define the data layer.
"""

import re
from enum import Enum, auto
from html import escape
from typing import Literal
from urllib.parse import urlparse

import networkx as nx
from lancedb.pydantic import LanceModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field, field_serializer

from .kg.types import GraphV2
from .utils import PALLETES

__all__ = [
    "SearchMethod",
    "GraphParameters",
    "Node",
    "Edge",
    "Graph",
    "SourceRecord",
    "DocumentRecord",
    "Document",
    "Chunk",
    "Message",
    "AssistantResponse",
]

MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]*)\)")


def _html_encode_text(value: str | None) -> str:
    """
    Encode model-controlled text for safe insertion into HTML clients.
    """
    neutralized = MARKDOWN_IMAGE_RE.sub(
        lambda match: f"[image omitted: {(match.group(1) or '').strip()}]",
        value or "",
    )
    return escape(neutralized, quote=True)


def _safe_href(value: str | None) -> str:
    """
    Restrict streamed links to ordinary web URLs.
    """
    url = (value or "").strip()
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return ""
    return url


def _html_encode_json_strings(value):
    if isinstance(value, str):
        return _html_encode_text(value)
    if isinstance(value, list):
        return [_html_encode_json_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: _html_encode_json_strings(item) for key, item in value.items()}
    return value


class SearchMethod(Enum):
    """
    Search method for graph retrieval.
    """

    EXACT = auto()
    VECTOR = auto()


class SharedParameters(BaseModel):
    """
    Shared query parameters for the underlying knowledge graph.
    """

    hops: int = Field(
        default=2,
        ge=0,
        description="Number of hops to extract nodes from."
        "For example, 1 means extract the central node and the immediate neighbours"
        " (1-hop neighbourhood).",
    )


class GraphParameters(SharedParameters):
    """
    Query parameters for `/graph` endpoint.
    """

    query: str = Field(
        description="A query to retrieve a knowledge graph for",
        min_length=2,
        json_schema_extra={"example": "climate change mitigation"},
    )


class Node(BaseModel):
    """
    A node in the knowledge graph representing a single concept.
    """

    name: str = Field(
        description="Name of the node",
        examples=["climate change mitigation", "energy policy"],
    )
    description: str | None = Field(
        description="Description of the node",
        examples=[
            "Actions to limit global warming by reducing greenhouse gas emissions.",
            "Rules managing energy use and resources sustainably.",
        ],
    )
    neighbourhood: int = Field(
        default=0,
        description="""Numeric value for a k-hop neighbourhood, with designations as follows:
        0 for the central node(s), 1 for secondary nodes and so on""",
        examples=[0, 2],
        ge=0,
    )
    weight: float = Field(
        default=0.0,
        description="Numeric value indicating the node's relevance",
        examples=[2.71828, 3.14159],
        ge=0.0,
    )
    colour: str = Field(
        default="#A9B1B7",
        description="Hex colour value for the node",
        examples=["#3288CE", "#55606E"],
    )
    vector: list[float] | None = Field(
        default=None,
        description="Vector embedding of the node",
        exclude=True,
    )

    def __hash__(self) -> int:
        return self.name.__hash__()

    def __lt__(self, other):
        if isinstance(other, Node):
            return self.name < other.name
        return NotImplemented


class Edge(BaseModel):
    """
    An edge in the knowledge graph representing a relation between nodes.

    Note that a node for `subject` property is guaranteed to exists, but the `object`
    value might not have a corresponding node entity in the graph.
    """

    subject: str = Field(
        description="Name of the node the edge goes from",
        examples=["climate change mitigation"],
    )
    predicate: str = Field(
        description="Name of the edge relation",
        examples=["addresses"],
    )
    object: str = Field(
        description="Name of the node the edge goes to",
        examples=["energy policy"],
    )
    description: str = Field(
        description="Description of the edge",
        examples=[
            "Climate adaptation strategies address the impacts of extreme weather"
            " events on communities and infrastructure.",
            "Climate adaptation strategies support community engagement to ensure"
            " inclusive decision-making and implementation.",
        ],
    )
    weight: float = Field(
        default=1.0,
        description="Numeric value indicating the edge's importance",
        examples=[2.71828, 3.14159],
        ge=0.0,
    )
    level: int = Field(
        default=1,
        description="Hop distance from the central node (1 = direct connection, 2 = 2-hop, etc.)",
        examples=[1, 2, 3],
        ge=1,
    )

    def __hash__(self) -> int:
        return (self.subject + self.predicate + self.object).__hash__()


class Graph(BaseModel, frozen=True):
    """
    Knowledge graph centred around a single concept.
    """

    nodes: list[Node] = Field(description="A list of nodes in the graph.")
    edges: list[Edge] = Field(description="A list of edges in the graph.")

    def __add__(self, other: "Graph") -> "Graph":
        if not isinstance(other, Graph):
            raise NotImplementedError
        nodes = set(self.nodes) | set(other.nodes)
        edges = set(self.edges) | set(other.edges)
        return Graph(nodes=nodes, edges=edges)

    @classmethod
    def from_networkx(cls, graph: nx.Graph, sources: str | list[str]) -> "Graph":
        """
        Create a graph response from a NetworkX graph.

        Parameters
        ----------
        graph : nx.Graph
            Abitrary NetworkX graph.
        sources : str | list[str]
            Name(s) of the central node(s) to define neighbourhood.

        Returns
        -------
        Graph
            Response graph with colour-coded nodes based on the neighbourhood
            around the source nodes.
        """
        sources = [sources] if isinstance(sources, str) else sources
        # mark all source nodes as central (neighbourhood=0)
        for source in sources:
            graph.nodes[source]["neighbourhood"] = 0
            graph.nodes[source]["colour"] = "#9F7DC5"
        # compute minimum neighbourhood from any source using undirected paths
        undirected = graph.to_undirected()
        for source in sources:
            for target, (path, *_) in nx.single_source_all_shortest_paths(
                undirected, source=source
            ):
                hop = len(path) - 1
                if hop == 0:
                    continue  # skip source nodes
                # keep the minimum neighbourhood value across all sources
                current = graph.nodes[target].get("neighbourhood")
                if current is None or hop < current:
                    graph.nodes[target]["neighbourhood"] = hop
                    # colours for nodes are determined by their neighbourhood
                    if hop <= len(PALLETES[0]):
                        index = sum(map(ord, target)) % len(PALLETES)
                        graph.nodes[target]["colour"] = PALLETES[index][hop - 1]
        return cls(
            nodes=[{"name": name} | data for name, data in graph.nodes(data=True)],
            edges=[
                {
                    "subject": subject,
                    "object": object,
                    "level": graph.nodes[subject].get("neighbourhood", 0) + 1,
                }
                | data
                for subject, object, data in graph.edges(data=True)
            ],
        )


class Document(LanceModel):
    """
    Publication document.
    """

    document_id: str | None = Field(
        default=None, description="Stable document identifier"
    )
    source: str | None = Field(
        default=None, description="Document source identifier or name"
    )
    publisher: str | None = Field(default=None, description="Publishing organization")
    title: str = Field(description="Document title if available")
    year: int = Field(description="Publication year if available")
    language: str = Field(description="Document language")
    url: str = Field(description="URL to the source document")
    summary: str | None = Field(description="Brief document summary if available")
    document_type: str | None = Field(
        default=None, description="Document type classification"
    )
    publication_date: str | None = Field(
        default=None, description="ISO publication date if available"
    )
    series_name: str | None = Field(
        default=None, description="Series or report family name"
    )
    topics: list[str] | None = Field(default=None, description="Topical tags")
    geographies: list[str] | None = Field(default=None, description="Geographic tags")

    def __hash__(self) -> int:
        """
        Allows Document instances to be hashable based on the document identity.
        """
        return hash(self.document_id or self.url or self.title)

    def __eq__(self, other: "Document") -> bool:
        """
        Compares two Document instances for equality based on stable identity where possible.
        """
        if not isinstance(other, Document):
            return NotImplemented
        left = self.document_id or self.url or self.title
        right = other.document_id or other.url or other.title
        return left == right

    def __lt__(self, other: "Document") -> bool:
        """
        Less-than comparison for sorting by year first, then title.
        """
        if self.year != other.year:
            # descending order by year
            return self.year > other.year
        # alphabetically by title (if years are the same)
        return self.title < other.title

    def to_stream_payload(self) -> dict[str, str | int | None]:
        """
        Convert the document to the compact streamed reference shape expected by clients.
        """
        year = self.year if isinstance(self.year, int) and self.year > 0 else None
        return {
            "title": _html_encode_text(self.title),
            "year": year,
            "language": _html_encode_text(self.language),
            "url": _safe_href(self.url),
            "summary": _html_encode_text(self.summary),
        }


class Chunk(Document):
    """
    Chunk of a publication document.
    """

    content: str = Field(description="Text content of a chunk")
    chunk_id: str | None = Field(default=None, description="Stable chunk identifier")
    chunk_index: int | None = Field(
        default=None, description="Chunk order within a document"
    )
    content_type: str | None = Field(default=None, description="Chunk content type")
    section_title: str | None = Field(
        default=None, description="Section heading if available"
    )
    page_start: int | None = Field(
        default=None, description="First page covered by the chunk"
    )
    page_end: int | None = Field(
        default=None, description="Last page covered by the chunk"
    )
    token_count: int | None = Field(default=None, description="Estimated token count")
    chunk_summary: str | None = Field(
        default=None, description="Optional short chunk summary"
    )

    def to_context(self) -> dict:
        """
        Convert a chunk to a simple context to be fed to a GenAI model.
        """
        return {
            "document_id": self.document_id,
            "title": self.title,
            "year": self.year,
            "summary": self.summary,
            "section_title": self.section_title,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "content_type": self.content_type,
            "chunk_id": self.chunk_id,
            "content": self.content,
        }


class SourceRecord(LanceModel):
    """
    Canonical metadata for a source registry entry.
    """

    source_id: str
    name: str
    organization: str
    authority_tier: str = Field(default="partner")
    license_policy: str | None = None
    robots_policy: str | None = None
    base_url: str | None = None
    ingestion_method: str = Field(default="manual_manifest")
    enabled: bool = Field(default=True)
    review_policy: str = Field(default="hybrid_editorial")
    created_at: str | None = None
    updated_at: str | None = None


class DocumentRecord(LanceModel):
    """
    Canonical publication-level record stored in the retrieval corpus.
    """

    document_id: str
    source_id: str
    canonical_title: str
    url: str
    language: str
    document_type: str
    publication_date: str | None = None
    year: int = 0
    summary: str | None = None
    status: str = Field(default="approved")
    ingested_at: str | None = None
    updated_at: str | None = None
    content_hash: str | None = None
    parser_version: str | None = None
    embedding_version: str | None = None
    subtitle: str | None = None
    authors: list[str] | None = None
    publisher: str | None = None
    series_name: str | None = None
    series_id: str | None = None
    country_codes: list[str] | None = None
    region_codes: list[str] | None = None
    topic_tags: list[str] | None = None
    sdg_tags: list[str] | None = None
    sector_tags: list[str] | None = None
    audience_tags: list[str] | None = None
    source_priority: float | None = None
    quality_score: float | None = None
    dedupe_group_id: str | None = None
    is_flagship: bool = False
    is_data_report: bool = False
    has_tables: bool | None = None
    has_figures: bool | None = None
    page_count: int | None = None
    review_notes: str | None = None
    topic_tags_text: str | None = None
    geography_tags_text: str | None = None
    audience_tags_text: str | None = None
    project_ids: list[str] | None = None
    project_numbers: list[str] | None = None
    data_classification: str | None = None
    publication_status: str | None = None
    review_state: str | None = None
    review_required: bool | None = None
    sensitive_content_flags: list[str] | None = None
    validation_category: str | None = None
    classifier_version: str | None = None
    ruleset_version: str | None = None
    ruleset_revision: str | None = None
    policy_version: str | None = None
    source_sha256: str | None = None


class Message(BaseModel):
    """
    Simple message for handling conversations with a chatbot.
    """

    role: Literal["assistant", "human"] = Field(
        description="The actor the message belongs to",
        examples=["human"],
    )
    content: str = Field(
        description="Text content of the message",
        min_length=0,
        max_length=16_384,
        examples=[
            "How does climate change adaptation differ from climate change mitigation?"
        ],
    )

    def to_langchain(self) -> AIMessage | HumanMessage:
        """
        Convert the message to a langchain-compatible class.

        Returns
        -------
        AIMessage | HumanMessage
            Langchain message class for AI or human.
        """
        return (
            AIMessage(self.content)
            if self.role == "assistant"
            else HumanMessage(self.content)
        )


class AssistantResponse(Message):
    """
    Assistant message from the LLM, optionally with relevant knowledge subgraphs.
    """

    ideas: list[str] | None = Field(
        default=None,
        description="A list of relevant query ideas based on the user message",
    )
    documents: list[Document] | None = Field(
        default=None, description="One or more documents relevant to the user message"
    )
    graph: Graph | GraphV2 | None = Field(default=None)

    @field_serializer("content")
    def _serialise_content(self, content: str) -> str:
        return _html_encode_text(content)

    @field_serializer("ideas")
    def _serialise_ideas(self, ideas: list[str] | None) -> list[str] | None:
        if ideas is None:
            return None
        return [_html_encode_text(idea) for idea in ideas]

    @field_serializer("documents")
    def _serialise_documents(
        self, documents: list[Document] | None
    ) -> list[dict[str, str | int | None]] | None:
        if documents is None:
            return None
        return [document.to_stream_payload() for document in documents]

    @field_serializer("graph")
    def _serialise_graph(self, graph: Graph | GraphV2 | None):
        if graph is None:
            return None
        return _html_encode_json_strings(graph.model_dump())

    def clear(self) -> None:
        """
        Set non-message properties to `None` to reduce payload size.
        """
        self.ideas = None
        self.documents = None
        self.graph = None
