"""Unit tests for HelixMind settings and configuration."""

from pathlib import Path
from helixmind.config import Settings, get_settings, save_env_variables, update_settings


def test_settings_default_values():
    settings = Settings()
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_timeout > 0
    assert "llama" in settings.ollama_default_model.lower()
    assert settings.default_max_hypotheses > 0


def test_get_settings_singleton():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    assert s1.cache_dir.exists()
    assert s1.data_dir.exists()


def test_save_env_variables_and_update(tmp_path: Path):
    test_env = tmp_path / ".env.test"
    test_env.write_text("OLLAMA_BASE_URL=http://localhost:11434\nOLLAMA_DEFAULT_MODEL=llama3.2:1b\n")

    save_env_variables({"OLLAMA_DEFAULT_MODEL": "mistral:7b", "ENABLE_DEPMAP": "False"}, env_path=test_env)
    content = test_env.read_text()
    assert "OLLAMA_DEFAULT_MODEL=mistral:7b" in content
    assert "ENABLE_DEPMAP=False" in content

    # Test update_settings without mutating global test environment permanently
    s = update_settings({"OLLAMA_DEFAULT_MODEL": "mistral:7b"}, persist=False)
    assert s.ollama_default_model == "mistral:7b"
    # Reset back to default
    update_settings({"OLLAMA_DEFAULT_MODEL": "llama3.2:1b"}, persist=False)
