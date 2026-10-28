"""`simic doctor`: each check's outcomes, and the command's verdict."""

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError
from pydantic import SecretStr

from simic import _doctor
from simic.cli import cli
from simic.settings import Settings


def _run(stdout: str = "", returncode: int = 0):
    return SimpleNamespace(stdout=stdout, returncode=returncode)


# --- tools ---


def test_tools_missing(tmp_vault):
    with patch("simic._doctor.shutil.which", side_effect=lambda t: None if t == "pactl" else t):
        check = _doctor._tools(Settings(vault_path=tmp_vault))
    assert check.outcome == "fail"
    assert check.detail.startswith("missing pactl — ")


@pytest.mark.parametrize(
    ("encoders", "outcome"),
    [(" A..... libmp3lame  MP3 (MPEG audio layer 3)", "ok"), (" A..... aac  AAC", "fail")],
)
def test_tools_need_an_mp3_encoder_for_mp3_capture(tmp_vault, encoders, outcome):
    with (
        patch("simic._doctor.shutil.which", side_effect=lambda t: t),
        patch("simic._doctor.subprocess.run", return_value=_run(encoders)),
    ):
        assert _doctor._tools(Settings(vault_path=tmp_vault)).outcome == outcome


def test_tools_for_wav_capture_do_not_ask_ffmpeg(tmp_vault):
    with (
        patch("simic._doctor.shutil.which", side_effect=lambda t: t),
        patch("simic._doctor.subprocess.run") as run,
    ):
        check = _doctor._tools(Settings(vault_path=tmp_vault, audio_format="wav"))
    assert (check.outcome, check.detail) == ("ok", "parecord, pactl, ffmpeg")
    run.assert_not_called()


# --- sound server and sources ---


_PACTL_INFO = (
    "Server String: /run/user/1000/pulse/native\nServer Name: PulseAudio (on PipeWire 1.2.7)\n"
)
_NO_SERVER = "no PulseAudio/PipeWire server answers — is the session's audio service running?"


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (_run(_PACTL_INFO), ("ok", "PulseAudio (on PipeWire 1.2.7)")),
        (_run("Server String: x\n"), ("ok", "reachable")),
        (_run("", returncode=1), ("fail", _NO_SERVER)),
    ],
)
def test_sound_server(tmp_vault, result, expected):
    with (
        patch("simic._doctor.shutil.which", return_value="/usr/bin/pactl"),
        patch("simic._doctor.subprocess.run", return_value=result),
    ):
        check = _doctor._sound_server(Settings(vault_path=tmp_vault))
    assert (check.outcome, check.detail) == expected


def test_sound_server_without_pactl(tmp_vault):
    with patch("simic._doctor.shutil.which", return_value=None):
        assert _doctor._sound_server(Settings(vault_path=tmp_vault)).outcome == "fail"


def test_sources(tmp_vault):
    settings = Settings(
        vault_path=tmp_vault, monitor_source="speakers.monitor", mic_source="usb-mic"
    )
    check = _doctor._sources(settings)
    assert (check.outcome, check.detail) == ("ok", "monitor speakers.monitor, mic usb-mic")


def test_sources_not_found(tmp_vault):
    with patch(
        "simic._doctor.detect_devices",
        side_effect=subprocess.CalledProcessError(1, ["pactl"]),
    ):
        assert _doctor._sources(Settings(vault_path=tmp_vault)).outcome == "fail"


def test_echo_cancellation_loads_and_is_let_go(tmp_vault, fake_pw_cli):
    check = _doctor._echo_cancel(Settings(vault_path=tmp_vault))

    assert (check.outcome, check.detail) == ("ok", "PipeWire WebRTC canceller on the mic")
    [canceller] = fake_pw_cli()
    assert canceller.stopped_by == "TERM"


def test_echo_cancellation_that_does_not_load_is_a_warning(tmp_vault, fake_parecord):
    """isolate_pipewire's pw-cli: PipeWire without the WebRTC plugin."""
    check = _doctor._echo_cancel(Settings(vault_path=tmp_vault))

    assert (check.outcome, check.detail) == (
        "warn",
        "PipeWire's echo-cancel module did not load (is the WebRTC AEC plugin installed?)"
        " — on speakers, use headphones",
    )


def test_echo_cancellation_turned_off(tmp_vault):
    check = _doctor._echo_cancel(Settings(vault_path=tmp_vault, echo_cancel=False))

    assert (check.outcome, check.detail) == (
        "ok",
        "off — on speakers the mic records the call too",
    )


def test_echo_cancellation_without_sources_is_a_warning(tmp_vault):
    with patch(
        "simic._doctor.detect_devices",
        side_effect=subprocess.CalledProcessError(1, ["pactl"]),
    ):
        check = _doctor._echo_cancel(Settings(vault_path=tmp_vault))

    assert check.outcome == "warn"


# --- vault and disk ---


def test_vault_states(tmp_path):
    missing = tmp_path / "nowhere"
    assert _doctor._vault(Settings(vault_path=missing)).outcome == "warn"
    assert _doctor._vault(Settings(vault_path=tmp_path)).outcome == "ok"
    a_file = tmp_path / "file"
    a_file.write_text("")
    assert _doctor._vault(Settings(vault_path=a_file)).outcome == "fail"


@pytest.mark.parametrize(
    ("free", "outcome", "hours"),
    [(1_039_999_999, "warn", 1), (1_040_000_000, "ok", 2), (52_000_000_000, "ok", 100)],
)
def test_disk_counts_hours_of_meetings(tmp_vault, free, outcome, hours):
    with patch("simic._doctor.shutil.disk_usage", return_value=SimpleNamespace(free=free)):
        check = _doctor._disk(Settings(vault_path=tmp_vault))
    assert check.outcome == outcome
    assert f"(about {hours} hours of meetings)" in check.detail


def test_disk_refuses_a_shared_sessions_dir(tmp_path):
    shared = tmp_path / "sessions"
    shared.mkdir()
    shared.chmod(0o755)
    check = _doctor._disk(Settings(vault_path=tmp_path, sessions_dir=shared))
    assert check.outcome == "fail"


# --- model and pending ---


def test_model_not_downloaded(tmp_vault):
    with patch("simic._doctor.download_model", side_effect=LocalEntryNotFoundError("x")):
        check = _doctor._model(Settings(vault_path=tmp_vault))
    assert check.outcome == "warn"
    assert check.detail.startswith("large-v3-turbo is not downloaded yet")


def test_model_partly_downloaded(tmp_vault, tmp_path):
    with patch("simic._doctor.download_model", return_value=str(tmp_path)):
        check = _doctor._model(Settings(vault_path=tmp_vault))
    assert check.outcome == "warn"


def test_model_ready(tmp_vault, tmp_path):
    (tmp_path / "model.bin").write_bytes(b"")
    with patch("simic._doctor.download_model", return_value=str(tmp_path)):
        check = _doctor._model(Settings(vault_path=tmp_vault))
    assert (check.outcome, check.detail) == ("ok", "large-v3-turbo on cpu")


def test_model_from_a_local_directory(tmp_vault, tmp_path):
    (tmp_path / "model.bin").write_bytes(b"")
    with patch("simic._doctor.download_model") as download:
        check = _doctor._model(Settings(vault_path=tmp_vault, stt_model=str(tmp_path)))
    assert check.outcome == "ok"
    download.assert_not_called()


def test_model_unknown_name(tmp_vault):
    with patch("simic._doctor.download_model", side_effect=ValueError("Invalid model size")):
        assert _doctor._model(Settings(vault_path=tmp_vault, stt_model="huge")).outcome == "fail"


def test_model_remote_backend(tmp_vault):
    check = _doctor._model(Settings(vault_path=tmp_vault, stt_backend="openai"))
    assert check.outcome == "ok"


def test_pending(tmp_vault, tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir(mode=0o700)
    settings = Settings(vault_path=tmp_vault, sessions_dir=sessions)
    assert _doctor._pending(settings).detail == "nothing waiting"
    (sessions / "2026-10-12_09-00-00").mkdir()
    check = _doctor._pending(settings)
    assert (check.outcome, check.detail) == (
        "warn",
        "not yet transcribed: 2026-10-12_09-00-00 — run 'simic recover'",
    )


# --- speakers and summary ---


def _find_spec_without(*missing: str):
    """importlib's find_spec on a machine without these packages: a missing parent
    raises, a missing leaf returns None."""

    def find_spec(name: str):
        if name.split(".", maxsplit=1)[0] in missing:
            if "." in name:
                raise ModuleNotFoundError(f"No module named {name.split('.', maxsplit=1)[0]!r}")
            return None
        return SimpleNamespace(name=name)

    return find_spec


def test_speakers_off(tmp_vault):
    check = _doctor._speakers(Settings(vault_path=tmp_vault, diarize=False))
    assert (check.outcome, check.detail) == ("ok", "off")


def test_speakers_without_pyannote(tmp_vault):
    settings = Settings(vault_path=tmp_vault, hf_token=SecretStr("hf_x"))
    with patch("simic._doctor.importlib.util.find_spec", _find_spec_without("pyannote")):
        check = _doctor._speakers(settings)
    assert (check.outcome, check.detail) == (
        "warn",
        "pyannote is not installed — install si-mic[diarize]",
    )


@pytest.mark.parametrize(
    ("token", "outcome", "detail"),
    [("", "warn", "no Hugging Face token — set SIMIC_HF_TOKEN"), ("hf_x", "ok", "pyannote")],
)
def test_speakers_need_a_token(tmp_vault, token, outcome, detail):
    settings = Settings(vault_path=tmp_vault, hf_token=SecretStr(token))
    with patch("simic._doctor.importlib.util.find_spec", _find_spec_without()):
        check = _doctor._speakers(settings)
    assert (check.outcome, check.detail) == (outcome, detail)


def test_summary_off(tmp_vault):
    check = _doctor._summary(Settings(vault_path=tmp_vault, summarize=False))
    assert (check.outcome, check.detail) == ("ok", "off")


def test_summary_without_a_key(tmp_vault):
    check = _doctor._summary(Settings(vault_path=tmp_vault, llm_provider="gemini"))
    assert (check.outcome, check.detail) == (
        "warn",
        "no API key for gemini — notes are saved without a summary",
    )


def test_summary_names_every_provider_it_would_try(tmp_vault, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    settings = Settings(vault_path=tmp_vault, llm_api_key=SecretStr("sk-ant-test"))
    with patch("simic._doctor.importlib.util.find_spec", _find_spec_without()):
        check = _doctor._summary(settings)
    assert (check.outcome, check.detail) == ("ok", "anthropic, then gemini")


@pytest.mark.parametrize(("provider", "package"), [("anthropic", "anthropic"), ("groq", "openai")])
def test_summary_needs_the_first_providers_package(tmp_vault, provider, package):
    settings = Settings(vault_path=tmp_vault, llm_provider=provider, llm_api_key=SecretStr("key"))
    with patch("simic._doctor.importlib.util.find_spec", _find_spec_without(package)):
        check = _doctor._summary(settings)
    assert (check.outcome, check.detail) == (
        "warn",
        f"{package} is not installed — install si-mic[llm]",
    )


# --- the command ---


@pytest.mark.parametrize(("outcome", "exit_code"), [("warn", 0), ("fail", 1)])
def test_doctor_prints_every_check_and_fails_only_on_failure(runner, vault_env, outcome, exit_code):
    checks = [
        _doctor.Check("Tools", "ok", "parecord, pactl, ffmpeg with MP3"),
        _doctor.Check("Model", outcome, "large-v3-turbo is not downloaded yet"),
    ]
    with patch("simic._doctor.run_checks", return_value=checks):
        result = runner.invoke(cli, ["doctor"])

    assert result.exit_code == exit_code
    mark = {"warn": "!", "fail": "✗"}[outcome]
    assert result.output == (
        "✓ Tools  parecord, pactl, ffmpeg with MP3\n"
        f"{mark} Model  large-v3-turbo is not downloaded yet\n"
    )
