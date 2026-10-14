"""Regression: a session another process is already transcribing was announced on the
desktop as "was not transcribed".

`process_session` notified every exception as a failure, SessionBusy included. With
background processing a second `simic recover` — or the detached run racing one
started by hand — refuses the session correctly and then told the user their meeting
was lost, while the other process was about to deliver its note.
"""

import pytest

from simic.pipeline import process_session
from simic.recorder import SessionBusy, processing, session_root
from simic.settings import Settings


def test_a_session_already_being_processed_is_not_reported_as_failed(vault_env, fake_notify_send):
    settings = Settings()
    session = session_root(settings) / "2026-10-13_09-00-00"
    session.mkdir()
    (session / "monitor.mp3").write_bytes(b"")
    (session / "mic.mp3").write_bytes(b"")

    with processing(session), pytest.raises(SessionBusy):
        process_session(session, settings)

    assert fake_notify_send() == []
