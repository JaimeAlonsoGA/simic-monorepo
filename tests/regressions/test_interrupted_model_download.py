"""Regression: an interrupted first model download broke tapeback until the cache was
deleted by hand.

Bug: the model is loaded from the local cache first, falling back to a download only on
LocalEntryNotFoundError. A download cut short (Ctrl+C, a closed laptop, 1.6 GB on a slow
line) leaves a snapshot directory without model.bin: the cache lookup finds the
snapshot, so no download is attempted, and ctranslate2 fails on the missing file —
on every run after, with "Unable to open file 'model.bin'".
"""

from unittest.mock import MagicMock, patch

import pytest

from tapeback.transcriber import Transcriber


def test_an_incomplete_cached_model_is_downloaded_again(settings):
    instance = MagicMock()
    incomplete = RuntimeError(
        "Unable to open file 'model.bin' in model '/hf/snapshots/0a363e9161cbc7ed1431c9597a8ceaf0'"
    )
    with patch("tapeback.transcriber.WhisperModel", side_effect=[incomplete, instance]) as model:
        transcriber = Transcriber(settings)

    assert [call.kwargs["local_files_only"] for call in model.call_args_list] == [True, False]
    assert transcriber._model is instance


def test_a_cuda_failure_is_not_mistaken_for_an_incomplete_download(settings):
    """A CUDA error is the CPU fallback's to handle; downloading again would not help."""
    settings = settings.model_copy(update={"device": "cuda"})
    instance = MagicMock()
    cuda = RuntimeError("CUDA failed with error out of memory")
    with (
        patch("tapeback.transcriber.WhisperModel", side_effect=[cuda, instance]) as model,
        patch("tapeback.transcriber.free_gpu_memory"),
        patch("tapeback.transcriber.preload_cuda_libs"),
    ):
        transcriber = Transcriber(settings)

    assert [(c.kwargs["device"], c.kwargs["local_files_only"]) for c in model.call_args_list] == [
        ("cuda", True),
        ("cpu", True),
    ]
    assert transcriber._model is instance


def test_a_failed_repair_reports_both_failures(settings):
    """Offline with a broken cache: the download's error, chained to the cache's."""
    incomplete = RuntimeError("Unable to open file 'model.bin' in model '/hf/snapshots/x'")
    offline = OSError("We couldn't connect to 'https://huggingface.co'")
    with (
        patch("tapeback.transcriber.WhisperModel", side_effect=[incomplete, offline]),
        pytest.raises(OSError, match="couldn't connect") as raised,
    ):
        Transcriber(settings)

    assert raised.value.__cause__ is incomplete
