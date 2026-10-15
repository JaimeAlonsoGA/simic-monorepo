"""Regression: `simic stop` failed on python-build-standalone, the interpreter of the
.deb and of `uv tool install`.

The background transcription was started with os.posix_spawn(setsid=True). That needs
POSIX_SPAWN_SETSID compiled in, and python-build-standalone is built against a glibc
without it: the call raised NotImplementedError, and `stop` ended in a traceback with
the meeting untranscribed. CI ran only on a distro Python, which has it, so nothing
saw this before a user would have.
"""

import os

import pytest

from simic._background import process_in_background
from simic.recorder import session_root
from simic.settings import Settings

SESSION = "2026-10-14_09-00-00"


@pytest.mark.usefixtures("interpreter_without_spawn_setsid", "vault_env")
def test_background_transcription_starts_without_spawn_setsid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    session = session_root(Settings()) / SESSION
    session.mkdir()

    pid = process_in_background(session, diarize=False, do_summarize=False)
    child_session = os.getsid(pid)
    os.waitpid(pid, 0)

    assert child_session == pid
