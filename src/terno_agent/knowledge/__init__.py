"""Knowledge-extraction pipeline.

Four phases run in parallel against a connected database to produce
a queryable knowledge base: organization context, schema crawl with
PII/profiling/relationships/enums/staleness, semantic annotation
(descriptions + value embeddings + gotchas), and validated example
question/SQL pairs.

Public surface mirrors the layering:

  - Agent / Runner:    KnowledgeExtractionAgent, KnowledgeRunner
  - Phases / Tasks:    Phase, Task, TaskResult, TaskStatus, PhaseResult
  - Context / Store:   PhaseContext, TaskContext, KnowledgeStore, InMemoryStore
  - User input:        PromptChannel, UserPrompt, PromptOption, UserResponse
"""

from terno_agent.knowledge.agent import KnowledgeExtractionAgent
from terno_agent.knowledge.base import (
    Phase,
    PhaseResult,
    Task,
    TaskResult,
    TaskStatus,
)
from terno_agent.knowledge.context import PhaseContext, TaskContext
from terno_agent.knowledge.prompts import (
    PromptChannel,
    PromptOption,
    UserPrompt,
    UserResponse,
)
from terno_agent.knowledge.runner import KnowledgeReport, KnowledgeRunner
from terno_agent.knowledge.store import InMemoryStore, KnowledgeStore

__all__ = [
    "InMemoryStore",
    "KnowledgeExtractionAgent",
    "KnowledgeReport",
    "KnowledgeRunner",
    "KnowledgeStore",
    "Phase",
    "PhaseContext",
    "PhaseResult",
    "PromptChannel",
    "PromptOption",
    "Task",
    "TaskContext",
    "TaskResult",
    "TaskStatus",
    "UserPrompt",
    "UserResponse",
]
