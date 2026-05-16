import os
import textwrap

from terno_agent.config import Config, _default_env_path, load_env


def _clear_caches():
    _default_env_path.cache_clear()


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
    _clear_caches()
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

    _clear_caches()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TERNO_LLM_PROVIDER", "anthropic")

    Config.from_env()
    assert os.environ["TERNO_LLM_PROVIDER"] == "anthropic"


def test_explicit_path_bypasses_cache(tmp_path, monkeypatch):
    env_a = tmp_path / "a.env"
    env_b = tmp_path / "b.env"
    env_a.write_text("TERNO_FROM=a\n")
    env_b.write_text("TERNO_FROM=b\n")

    monkeypatch.delenv("TERNO_FROM", raising=False)
    _clear_caches()

    assert load_env(env_a) == env_a
    assert os.environ["TERNO_FROM"] == "a"

    # Default (override=False) does not overwrite.
    load_env(env_b)
    assert os.environ["TERNO_FROM"] == "a"

    # override=True applies the new file.
    load_env(env_b, override=True)
    assert os.environ["TERNO_FROM"] == "b"
