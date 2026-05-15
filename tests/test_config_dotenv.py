import os
import textwrap

from terno_agent import config as config_mod
from terno_agent.config import Config, load_env


def test_load_env_reads_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        textwrap.dedent(
            """
            TERNO_LLM_PROVIDER=openai
            TERNO_LLM_MODEL=gpt-test
            TERNO_DATABASE_URL=sqlite:///./from-dotenv.db
            TERNO_SANDBOX=local
            OPENAI_API_KEY=sk-from-dotenv
            """
        ).strip()
    )

    for key in (
        "TERNO_LLM_PROVIDER",
        "TERNO_LLM_MODEL",
        "TERNO_LLM_API_KEY",
        "TERNO_DATABASE_URL",
        "TERNO_SANDBOX",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config_mod, "_DOTENV_LOADED", False)
    monkeypatch.chdir(tmp_path)

    loaded = load_env()
    assert loaded == env
    assert os.environ["TERNO_LLM_PROVIDER"] == "openai"

    cfg = Config.from_env()
    assert cfg.llm_provider == "openai"
    assert cfg.llm_model == "gpt-test"
    assert cfg.llm_api_key == "sk-from-dotenv"
    assert cfg.database_url == "sqlite:///./from-dotenv.db"
    assert cfg.sandbox == "local"


def test_process_env_overrides_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("TERNO_LLM_PROVIDER=openai\n")

    monkeypatch.setattr(config_mod, "_DOTENV_LOADED", False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TERNO_LLM_PROVIDER", "anthropic")

    Config.from_env()
    assert os.environ["TERNO_LLM_PROVIDER"] == "anthropic"
