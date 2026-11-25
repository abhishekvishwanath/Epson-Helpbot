import pytest

from app.core.config import REPO_ROOT, Settings


def test_relative_paths_resolve_against_repo_root():
    s = Settings(_env_file=None)
    assert s.chroma_persist_directory == (REPO_ROOT / "data/chroma_db").resolve()
    assert s.database_url.endswith(str(REPO_ROOT / "data" / "sessions.db"))


def test_cors_origins_accept_comma_separated():
    s = Settings(_env_file=None, cors_origins="http://a.com, http://b.com")
    assert s.cors_origins == ["http://a.com", "http://b.com"]


def test_rejects_target_above_max():
    with pytest.raises(ValueError):
        Settings(_env_file=None, chunk_target_tokens=1000, chunk_max_tokens=900)


def test_production_requires_admin_key():
    with pytest.raises(ValueError):
        Settings(_env_file=None, environment="production", admin_api_key="")


def test_env_file_with_comma_separated_origins(tmp_path):
    env = tmp_path / ".env"
    env.write_text("CORS_ORIGINS=http://a.com,http://b.com\nRETRIEVAL_TOP_K=7\n")
    s = Settings(_env_file=env)
    assert s.cors_origins == ["http://a.com", "http://b.com"] and s.retrieval_top_k == 7
