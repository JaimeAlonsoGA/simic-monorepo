"""Remote speech-to-text backends.

Local transcription lives in ``transcriber.py``. Remote backends plug in here
via ``load_remote_transcriber``. Today the only supported remote backend is
OpenAI (`SIMIC_STT_BACKEND=openai`); the factory is the seam for more later.
"""

from __future__ import annotations

from simic._lazy import TranscriberLike
from simic._stt_openai import OpenAITranscriber
from simic.settings import Settings


def load_remote_transcriber(settings: Settings) -> TranscriberLike:
    """Instantiate the configured remote STT backend."""
    if settings.stt_backend == "openai":
        return OpenAITranscriber(settings)

    raise RuntimeError(
        f"Unsupported remote STT backend {settings.stt_backend!r}. "
        "Supported: 'openai'. Local transcription uses SIMIC_STT_BACKEND=local."
    )
