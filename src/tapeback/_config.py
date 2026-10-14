"""The settings `tapeback ui` offers, and how a change reaches the user's .env.

A short list on purpose: where the notes go, what is recorded, how it is transcribed,
whether speakers are told apart and the meeting summarised. Everything else stays a
TAPEBACK_* variable — the window writes the same file the CLI reads, so the two never
disagree, and a power-user line in that file survives every change made here.

Values travel as the strings a .env holds ("true", a path, a model id). A change is
validated as the Settings it would produce before anything is written: a combination
the CLI would refuse to load never reaches the file.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dotenv import dotenv_values, set_key, unset_key
from pydantic import SecretStr, ValidationError

from tapeback._sources import Source
from tapeback._stt_caps import (
    OPENAI_MODEL_GPT_DIARIZE,
    OPENAI_MODEL_GPT_TRANSCRIBE,
    OPENAI_MODEL_WHISPER_1,
)
from tapeback.settings import Settings

ENV_PREFIX = "TAPEBACK_"
_PRIVATE = 0o600
AUTO_SOURCE = "auto"

type FieldKind = Literal["switch", "choice", "folder", "text", "secret"]
type Values = Mapping[str, str]


@dataclass(frozen=True)
class Choice:
    value: str
    label: str


type ChoiceSource = Callable[[Values, Sequence[Source]], list[Choice]]


def _always(_values: Values) -> bool:
    return True


@dataclass(frozen=True)
class Field:
    name: str
    title: str
    kind: FieldKind
    choices: ChoiceSource | None = None
    visible: Callable[[Values], bool] = _always
    # Settings that stop making sense when this one changes, and return to their default.
    resets: tuple[str, ...] = ()


@dataclass(frozen=True)
class Section:
    title: str
    fields: tuple[Field, ...]
    # The `doctor` checks that say whether this part of a recording will work.
    checks: tuple[str, ...] = ()


def _fixed(*pairs: tuple[str, str]) -> ChoiceSource:
    choices = [Choice(value, label) for value, label in pairs]
    return lambda _values, _sources: choices


def _monitors(_values: Values, sources: Sequence[Source]) -> list[Choice]:
    found = [Choice(s.name, s.description) for s in sources if s.is_monitor]
    return [Choice(AUTO_SOURCE, "Default output"), *found]


def _microphones(_values: Values, sources: Sequence[Source]) -> list[Choice]:
    found = [Choice(s.name, s.description) for s in sources if not s.is_monitor]
    return [Choice(AUTO_SOURCE, "Default microphone"), *found]


_LOCAL_MODELS = (
    Choice("small", "Fast"),
    Choice("large-v3-turbo", "Balanced"),
    Choice("large-v3", "Accurate"),
)
_OPENAI_MODELS = (
    Choice(OPENAI_MODEL_WHISPER_1, "Whisper"),
    Choice(OPENAI_MODEL_GPT_TRANSCRIBE, "GPT Transcribe"),
    Choice(OPENAI_MODEL_GPT_DIARIZE, "GPT-4o with speakers"),
)


def _models(values: Values, _sources: Sequence[Source]) -> list[Choice]:
    return list(_OPENAI_MODELS if values["stt_backend"] == "openai" else _LOCAL_MODELS)


def _is(name: str, value: str) -> Callable[[Values], bool]:
    return lambda values: values[name] == value


SECTIONS: tuple[Section, ...] = (
    Section(
        "Notes",
        (
            Field("vault_path", "Vault", "folder"),
            Field("meetings_dir", "Folder", "text"),
            Field("keep_audio", "Keep the recording", "switch"),
            Field("notify", "Notify when ready", "switch"),
        ),
        checks=("Vault", "Recordings"),
    ),
    Section(
        "Audio",
        (
            Field("monitor_source", "What you hear", "choice", _monitors),
            Field("mic_source", "Microphone", "choice", _microphones),
        ),
        checks=("Tools", "Sound server", "Sources"),
    ),
    Section(
        "Transcription",
        (
            Field(
                "stt_backend",
                "Engine",
                "choice",
                _fixed(("local", "This computer"), ("openai", "OpenAI")),
                resets=("stt_model",),
            ),
            Field("stt_model", "Model", "choice", _models),
            Field("stt_api_key", "OpenAI key", "secret", visible=_is("stt_backend", "openai")),
            Field(
                "language",
                "Language",
                "choice",
                _fixed(
                    ("auto", "Detect"),
                    ("en", "English"),
                    ("es", "Spanish"),
                    ("ca", "Catalan"),
                    ("fr", "French"),
                    ("de", "German"),
                    ("it", "Italian"),
                    ("pt", "Portuguese"),
                    ("nl", "Dutch"),
                    ("ru", "Russian"),
                    ("zh", "Chinese"),
                    ("ja", "Japanese"),
                ),
            ),
            Field(
                "device",
                "Processor",
                "choice",
                _fixed(("cpu", "CPU"), ("cuda", "NVIDIA GPU")),
                visible=_is("stt_backend", "local"),
            ),
        ),
        checks=("Model",),
    ),
    Section(
        "Speakers",
        (
            Field("diarize", "Tell speakers apart", "switch"),
            Field("hf_token", "Hugging Face token", "secret", visible=_is("diarize", "true")),
        ),
        checks=("Speakers",),
    ),
    Section(
        "Summary",
        (
            Field("summarize", "Summarize", "switch"),
            Field(
                "llm_provider",
                "Provider",
                "choice",
                _fixed(
                    ("anthropic", "Anthropic"),
                    ("openai", "OpenAI"),
                    ("gemini", "Gemini"),
                    ("groq", "Groq"),
                    ("openrouter", "OpenRouter"),
                    ("deepseek", "DeepSeek"),
                    ("qwen", "Qwen"),
                ),
                visible=_is("summarize", "true"),
            ),
            Field("llm_api_key", "API key", "secret", visible=_is("summarize", "true")),
            Field(
                "mask_pii",
                "Hide emails and phone numbers",
                "switch",
                visible=_is("summarize", "true"),
            ),
        ),
        checks=("Summary",),
    ),
)

FIELDS: dict[str, Field] = {f.name: f for section in SECTIONS for f in section.fields}


def env_key(name: str) -> str:
    return f"{ENV_PREFIX}{name.upper()}"


def as_text(value: object) -> str:
    """A setting the way a .env spells it."""
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def current_values(settings: Settings) -> dict[str, str]:
    return {name: as_text(getattr(settings, name)) for name in FIELDS}


def choices_for(f: Field, values: Values, sources: Sequence[Source]) -> list[Choice]:
    """The options to offer, always including the value in effect: a device that is
    unplugged or a model set by hand is shown as it is, not silently replaced."""
    if f.choices is None:
        return []
    options = f.choices(values, sources)
    current = values[f.name]
    if current and all(option.value != current for option in options):
        options.append(Choice(current, current))
    return options


def locked(environ: Mapping[str, str]) -> set[str]:
    """Fields an environment variable sets: it outranks the .env, so a change written
    there would not take effect."""
    return {name for name in FIELDS if env_key(name) in environ}


def changes_for(name: str, value: str) -> dict[str, str | None]:
    """What one edit writes. None removes the line, returning the setting to its
    default — as an empty value does, so a cleared key leaves no empty line behind."""
    changes: dict[str, str | None] = {name: value or None}
    for reset in FIELDS[name].resets:
        changes[reset] = None
    return changes


def save(env_file: Path, changes: Mapping[str, str | None]) -> None:
    """Write the changes to the .env, where every command reads them.

    Raises ValidationError, and writes nothing, when the result would not load; OSError
    when the file cannot be written.
    """
    # Strings as the .env holds them; pydantic converts and validates them, as it does
    # when it reads the file.
    candidate: dict[str, Any] = {
        name: Settings.model_fields[name].default if value is None else value
        for name, value in changes.items()
    }
    Settings(**candidate)
    env_file.parent.mkdir(parents=True, exist_ok=True)
    written = dotenv_values(env_file) if env_file.exists() else {}
    for name, value in changes.items():
        if value is not None:
            set_key(env_file, env_key(name), value, quote_mode="always")
        elif env_key(name) in written:
            unset_key(env_file, env_key(name))
    # It holds API keys. set_key keeps the mode a file already had, and one made by hand
    # is usually readable by every user on the machine.
    if env_file.exists():
        env_file.chmod(_PRIVATE)


def describe(error: ValidationError) -> str:
    """The first reason a change was refused, without pydantic's framing."""
    return str(error.errors()[0]["msg"]).removeprefix("Value error, ")
