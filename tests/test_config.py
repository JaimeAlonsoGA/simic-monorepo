"""What `simic ui` offers to change, and how a change reaches the .env every command
reads."""

import pytest
from pydantic import SecretStr, ValidationError

from simic import _config
from simic._config import Choice
from simic._sources import Source
from simic.settings import Settings

SOURCES = [
    Source("alsa_output.analog-stereo.monitor", "Monitor of Built-in Audio"),
    Source("alsa_input.analog-stereo", "Built-in Audio"),
    Source("bluez_input.headset", "Headset"),
]


def _values(**overrides: str) -> dict[str, str]:
    return {**_config.current_values(Settings()), **overrides}


def _choices(name: str, **overrides: str) -> list[Choice]:
    return _config.choices_for(_config.FIELDS[name], _values(**overrides), SOURCES)


# --- the offer ---


def test_every_field_is_a_setting():
    assert set(_config.FIELDS) <= set(Settings.model_fields)


def test_the_offer_is_short():
    """Simple by construction: a power-user setting added here should be a decision."""
    assert list(_config.FIELDS) == [
        "vault_path",
        "meetings_dir",
        "keep_audio",
        "notify",
        "monitor_source",
        "mic_source",
        "stt_backend",
        "stt_model",
        "stt_api_key",
        "language",
        "device",
        "diarize",
        "hf_token",
        "summarize",
        "llm_provider",
        "llm_api_key",
        "mask_pii",
    ]


def test_every_llm_provider_settings_accept_is_offered():
    offered = [choice.value for choice in _choices("llm_provider")]
    assert offered == ["anthropic", "openai", "gemini", "groq", "openrouter", "deepseek", "qwen"]
    for provider in offered:
        assert Settings.model_validate({"llm_provider": provider}).llm_provider == provider


def test_models_follow_the_engine():
    assert [c.value for c in _choices("stt_model")] == ["small", "large-v3-turbo", "large-v3"]
    openai = _choices("stt_model", stt_backend="openai", stt_model="whisper-1")
    assert [c.value for c in openai] == [
        "whisper-1",
        "gpt-transcribe",
        "gpt-4o-transcribe-diarize",
    ]


def test_audio_offers_monitors_and_microphones_apart():
    assert _choices("monitor_source") == [
        Choice("auto", "Default output"),
        Choice("alsa_output.analog-stereo.monitor", "Monitor of Built-in Audio"),
    ]
    assert _choices("mic_source") == [
        Choice("auto", "Default microphone"),
        Choice("alsa_input.analog-stereo", "Built-in Audio"),
        Choice("bluez_input.headset", "Headset"),
    ]


def test_the_value_in_effect_is_offered_even_when_unknown():
    """An unplugged device or a model set by hand shows as it is, not replaced by the
    first option."""
    assert _choices("mic_source", mic_source="usb_mic")[-1] == Choice("usb_mic", "usb_mic")
    assert _choices("stt_model", stt_model="medium")[-1] == Choice("medium", "medium")


def test_a_known_value_is_offered_once():
    assert len(_choices("stt_model", stt_model="large-v3")) == 3


def test_a_switch_has_no_choices():
    assert _config.choices_for(_config.FIELDS["diarize"], _values(), SOURCES) == []


@pytest.mark.parametrize(
    ("name", "values", "visible"),
    [
        ("stt_api_key", {"stt_backend": "openai"}, True),
        ("stt_api_key", {"stt_backend": "local"}, False),
        ("device", {"stt_backend": "local"}, True),
        ("device", {"stt_backend": "openai"}, False),
        ("hf_token", {"diarize": "true"}, True),
        ("hf_token", {"diarize": "false"}, False),
        ("llm_api_key", {"summarize": "true"}, True),
        ("llm_api_key", {"summarize": "false"}, False),
        ("vault_path", {}, True),
    ],
)
def test_a_field_shows_only_where_it_applies(name, values, visible):
    assert _config.FIELDS[name].visible(_values(**values)) is visible


# --- values ---


def test_values_are_spelled_as_a_env_file_spells_them(tmp_path):
    settings = Settings(
        vault_path=tmp_path, keep_audio=True, diarize=False, hf_token=SecretStr("hf_secret")
    )
    values = _config.current_values(settings)
    assert values["vault_path"] == str(tmp_path)
    assert values["keep_audio"] == "true"
    assert values["diarize"] == "false"
    assert values["hf_token"] == "hf_secret"
    assert values["stt_model"] == "large-v3-turbo"


def test_none_is_spelled_empty():
    assert _config.as_text(None) == ""


def test_a_field_set_in_the_environment_is_locked():
    assert _config.locked({"SIMIC_DEVICE": "cuda", "SIMIC_BEAM_SIZE": "5"}) == {"device"}


def test_an_empty_value_returns_to_the_default():
    assert _config.changes_for("llm_api_key", "") == {"llm_api_key": None}


def test_a_new_engine_returns_the_model_to_its_default():
    """large-v3-turbo is no OpenAI model: kept, it would make the settings unloadable."""
    assert _config.changes_for("stt_backend", "openai") == {
        "stt_backend": "openai",
        "stt_model": None,
    }


# --- saving ---


def _read(env_file, monkeypatch) -> Settings:
    """Settings as the next command loads them, from this file alone."""
    monkeypatch.setitem(Settings.model_config, "env_file", (env_file,))
    return Settings()


def test_save_writes_the_env_file_settings_read(tmp_path, monkeypatch):
    env_file = tmp_path / "simic" / ".env"

    _config.save(env_file, {"vault_path": str(tmp_path / "My Notes"), "diarize": "false"})

    assert env_file.read_text() == (
        f"SIMIC_VAULT_PATH='{tmp_path}/My Notes'\nSIMIC_DIARIZE='false'\n"
    )
    loaded = _read(env_file, monkeypatch)
    assert loaded.vault_path == tmp_path / "My Notes"
    assert loaded.diarize is False


def test_save_keeps_a_new_file_private(tmp_path):
    """It holds API keys."""
    env_file = tmp_path / ".env"
    _config.save(env_file, {"llm_api_key": "sk-test"})
    assert oct(env_file.stat().st_mode & 0o777) == "0o600"


def test_save_makes_a_readable_file_private(tmp_path):
    """A .env made by hand is often 0644. Once it holds an API key, other users on the
    machine must not be able to read it."""
    env_file = tmp_path / ".env"
    env_file.write_text("SIMIC_DEVICE=cpu\n")
    env_file.chmod(0o644)

    _config.save(env_file, {"llm_api_key": "sk-test"})

    assert oct(env_file.stat().st_mode & 0o777) == "0o600"


def test_save_keeps_every_other_line(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("# mine\nSIMIC_BEAM_SIZE=5\nSIMIC_DEVICE=cpu\n")

    _config.save(env_file, {"device": "cuda"})

    assert env_file.read_text() == "# mine\nSIMIC_BEAM_SIZE=5\nSIMIC_DEVICE='cuda'\n"


def test_save_removes_a_setting_returned_to_its_default(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("SIMIC_STT_BACKEND=local\nSIMIC_STT_MODEL=large-v3\n")

    _config.save(env_file, _config.changes_for("stt_backend", "openai"))

    assert env_file.read_text() == "SIMIC_STT_BACKEND='openai'\n"
    assert _read(env_file, monkeypatch).stt_model == "whisper-1"


def test_save_leaves_a_missing_file_missing_when_there_is_nothing_to_write(tmp_path):
    env_file = tmp_path / ".env"
    _config.save(env_file, {"llm_api_key": None})
    assert not env_file.exists()


def test_save_refuses_settings_that_would_not_load_and_writes_nothing(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("SIMIC_DEVICE=cpu\n")

    with pytest.raises(ValidationError) as caught:
        _config.save(env_file, {"stt_backend": "openai", "stt_model": "large-v3"})

    assert env_file.read_text() == "SIMIC_DEVICE=cpu\n"
    assert _config.describe(caught.value).startswith("Unsupported OpenAI STT model 'large-v3'")


def test_save_writes_the_transcripts_folder(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"

    _config.save(env_file, _config.changes_for("meetings_dir", "Voice notes"))

    assert env_file.read_text() == "SIMIC_MEETINGS_DIR='Voice notes'\n"
    assert _read(env_file, monkeypatch).meetings_dir == "Voice notes"


def test_save_refuses_a_transcripts_folder_outside_the_vault(tmp_path):
    env_file = tmp_path / ".env"

    with pytest.raises(ValidationError) as caught:
        _config.save(env_file, _config.changes_for("meetings_dir", "../elsewhere"))

    assert not env_file.exists()
    assert _config.describe(caught.value) == "'../elsewhere' is not a folder inside the vault"
