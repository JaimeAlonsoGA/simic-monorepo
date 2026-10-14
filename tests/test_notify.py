"""Desktop notifications: sent when they can be, silently skipped when they cannot."""

import subprocess
from pathlib import Path
from unittest.mock import patch

from simic._notify import note_ready, notify
from simic.settings import Settings


def test_a_ready_note_is_announced_by_name(tmp_vault, fake_notify_send):
    note_ready(Settings(vault_path=tmp_vault), Path("/vault/meetings/2026-10-12_09-00-00.md"))

    assert fake_notify_send() == [("Meeting note ready", "2026-10-12_09-00-00.md")]


def test_notifications_can_be_turned_off(tmp_vault, fake_notify_send):
    notify(Settings(vault_path=tmp_vault, notify=False), "title", "body")

    assert fake_notify_send() == []


def test_without_notify_send_nothing_is_attempted(tmp_vault):
    with (
        patch("simic._notify.shutil.which", return_value=None),
        patch("simic._notify.subprocess.run") as run,
    ):
        notify(Settings(vault_path=tmp_vault, notify=True), "title", "body")

    run.assert_not_called()


def test_a_daemon_that_never_answers_does_not_hold_up_the_run(tmp_vault):
    with (
        patch("simic._notify.shutil.which", return_value="/usr/bin/notify-send"),
        patch(
            "simic._notify.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["notify-send"], 5.0),
        ),
    ):
        notify(Settings(vault_path=tmp_vault, notify=True), "title", "body")
