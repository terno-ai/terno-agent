"""Embedding client protocol and providers.

Parallel to `terno_agent.llm.base.LLMClient` — provider-agnostic interface
plus a small factory. Imports of provider SDKs are deferred so optional
dependencies stay optional.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from terno_agent.core.exceptions import ConfigError, TernoError


class EmbeddingError(TernoError):
    """Raised when an embedding call fails."""


@runtime_checkable
class EmbeddingClient(Protocol):
    """Produces dense vector embeddings for one or more texts."""

    model: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbeddingClient:
    """OpenAI Embeddings API client.

    Defaults to ``text-embedding-3-small`` (1536 dimensions, cheap). Texts
    are batched in chunks of 100 to stay well under the API's per-request
    cap.
    """

    BATCH_SIZE = 100

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ConfigError(
                "openai package not installed. Install with: pip install 'terno-agent[openai]'"
            ) from exc
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[start : start + self.BATCH_SIZE]
            try:
                response = self._client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
            except Exception as exc:
                raise EmbeddingError(f"OpenAI embeddings call failed: {exc}") from exc
            out.extend(item.embedding for item in response.data)
        return out


def create_embedding_client(
    provider: str = "openai",
    *,
    api_key: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> EmbeddingClient:
    """Return an `EmbeddingClient` for the requested provider."""
    provider = provider.lower().strip()
    if provider == "openai":
        return OpenAIEmbeddingClient(
            api_key=api_key,
            model=model or "text-embedding-3-small",
            **kwargs,
        )
    raise ConfigError(
        f"Unknown embedding provider: {provider!r}. Supported: openai."
    )


__all__ = [
    "EmbeddingClient",
    "EmbeddingError",
    "OpenAIEmbeddingClient",
    "create_embedding_client",
]
