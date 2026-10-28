"""Regression: `simic tray` told every GNOME Wayland user to install the AppIndicator
extension, even with the extension installed, enabled and showing the icon.

The hint was decided from XDG_SESSION_TYPE and XDG_CURRENT_DESKTOP alone, before the
tray had tried to register: a guess, printed on every start. Whether a tray host is
there is known only by asking for it — the StatusNotifierWatcher on the session bus,
which the extension provides. The hint now follows a failed registration, and only
then.
"""

import logging

import pytest

from tests.fixtures import requires_dbus

pytestmark = requires_dbus

from simic.tray import run_tray  # noqa: E402

GNOME_WAYLAND = {"XDG_SESSION_TYPE": "wayland", "XDG_CURRENT_DESKTOP": "ubuntu:GNOME"}


@pytest.fixture(autouse=True)
def gnome_wayland(monkeypatch):
    for name, value in GNOME_WAYLAND.items():
        monkeypatch.setenv(name, value)


@pytest.mark.usefixtures("vault_env")
def test_no_hint_when_the_watcher_is_there(session_bus, caplog):
    with caplog.at_level(logging.WARNING, logger="simic.tray"):
        run_tray()

    session_bus.registered.assert_awaited_once_with(":1.42")
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


@pytest.mark.usefixtures("vault_env")
def test_the_hint_once_when_no_watcher_answers(session_bus, caplog):
    session_bus.watcher_gone = True

    with caplog.at_level(logging.WARNING, logger="simic.tray"):
        run_tray()

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "sudo apt install gnome-shell-extension-appindicator" in warnings[0]
