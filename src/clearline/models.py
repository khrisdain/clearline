"""Core domain types: documents, chunks, the people asking, and what the agent returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Classification(IntEnum):
    """Canadian government security markings, lowest to highest."""

    UNCLASSIFIED = 0
    PROTECTED_A = 1
    PROTECTED_B = 2

    @classmethod
    def parse(cls, value: str | int | Classification) -> Classification:
        if isinstance(value, Classification):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[value.strip().upper().replace("-", "_").replace(" ", "_")]

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").title()


@dataclass(frozen=True)
class Principal:
    """Whoever is asking: their clearance, the groups they belong to and their app roles."""

    subject: str
    clearance: Classification = Classification.UNCLASSIFIED
    groups: frozenset[str] = frozenset()
    roles: frozenset[str] = frozenset()

    def can_read(self, classification: Classification, groups: frozenset[str]) -> bool:
        """Cleared to the document's level, and in one of its groups (if it names any)."""
        return classification <= self.clearance and (not groups or bool(groups & self.groups))

    @property
    def is_admin(self) -> bool:
        return "clearline-admin" in self.roles

    @property
    def is_auditor(self) -> bool:
        return "clearline-auditor" in self.roles


@dataclass
class Document:
    id: str
    title: str
    text: str
    classification: Classification = Classification.UNCLASSIFIED
    groups: frozenset[str] = frozenset()
    source: str = ""


@dataclass
class Chunk:
    id: str
    document_id: str
    title: str
    section: str
    text: str
    classification: Classification
    groups: frozenset[str]
    position: int
    embedding: list[float] = field(default_factory=list, repr=False)


@dataclass
class Hit:
    chunk: Chunk
    score: float


# --- Provider-neutral chat types --------------------------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


class Span(BaseModel):
    """A stretch of the answer, [start, end) in characters, supported by the numbered passages in `markers`."""

    start: int
    end: int
    text: str
    markers: list[int]


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    spans: list[Span] = field(default_factory=list)  # citations the model returned natively (Cohere), if any


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


# --- API schemas ------------------------------------------------------------------------------


class DocumentIn(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1, max_length=2_000_000)
    classification: str = "unclassified"
    groups: list[str] = Field(default_factory=list)
    source: str = ""


class DocumentOut(BaseModel):
    id: str
    title: str
    classification: str
    groups: list[str]
    chunks: int


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class Citation(BaseModel):
    marker: int
    document_id: str
    title: str
    section: str
    classification: str
    snippet: str


class Step(BaseModel):
    tool: str
    arguments: dict[str, Any]
    results: int
    retrieval: dict[str, int | bool] | None = None  # per-stage counts for a search: dense, keyword, fused, kept


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    steps: list[Step]
    grounded: bool
    latency_ms: int
    spans: list[Span] = Field(default_factory=list)
    coverage: float = 0.0  # share of the answer's characters inside a cited span
    citation_mode: Literal["native", "markers"] = "markers"
    model: str = ""
