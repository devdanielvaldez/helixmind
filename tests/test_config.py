"""Unit tests for HelixMind settings and configuration."""

from helixmind.config import Settings, get_settings


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
