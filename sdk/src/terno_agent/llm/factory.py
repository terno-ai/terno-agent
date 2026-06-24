from __future__ import annotations

from terno_agent.core.exceptions import ConfigError
from terno_agent.llm.base import LLMClient


def create_llm_client(
    provider: str,
    model: str,
    api_key: str | None = None,
    *,
    provisioner_url: str | None = None,
    app_version: str | None = None,
    request_source: str | None = None,
) -> LLMClient:
    """Return an `LLMClient` for the requested provider.

    Imports are deferred so users only need the SDK they actually use. The
    ``terno`` provider proxies every completion through the Terno
    provisioner server (see ``terno_llm.TernoLLMClient``); the provisioner
    keyword args are ignored by the direct providers.
    """
    provider = provider.lower().strip()
    if provider == "anthropic":
        from terno_agent.llm.anthropic_client import AnthropicClient

        return AnthropicClient(api_key=api_key, model=model)
    if provider == "openai":
        from terno_agent.llm.openai_client import OpenAIClient

        return OpenAIClient(api_key=api_key, model=model)
    if provider in ("terno", "provisioner"):
        from terno_agent.llm.terno_llm import TernoLLMClient

        return TernoLLMClient(
            api_key=api_key,
            model=model,
            provisioner_url=provisioner_url,
            app_version=app_version,
            request_source=request_source,
        )
    raise ConfigError(
        f"Unknown LLM provider: {provider!r}. Supported: anthropic, openai, terno."
    )
