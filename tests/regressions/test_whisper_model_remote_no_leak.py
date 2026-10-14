"""Regression: SIMIC_WHISPER_MODEL must not leak into remote STT backends."""

import pytest

from simic.settings import Settings


def test_whisper_model_ignored_for_openai_backend(monkeypatch, vault_env):
    """WHISPER_MODEL + openai backend (no STT_MODEL) must resolve to whisper-1."""
    monkeypatch.delenv("SIMIC_STT_MODEL", raising=False)
    monkeypatch.setenv("SIMIC_WHISPER_MODEL", "large-v3-turbo")
    monkeypatch.setenv("SIMIC_STT_BACKEND", "openai")

    with pytest.warns(FutureWarning, match="SIMIC_WHISPER_MODEL"):
        s = Settings()
    assert s.stt_backend == "openai"
    assert s.stt_model == "whisper-1"
