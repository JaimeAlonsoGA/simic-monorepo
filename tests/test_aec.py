"""PipeWire's echo canceller as `simic start` loads it: the command, the process that
holds it, and what happens where it cannot load."""

import os
import signal

import pytest

from simic import _aec
from tests.fixtures import process_state


def test_default_devices_leave_the_canceller_following_the_desktop():
    """@DEFAULT_SOURCE@ and @DEFAULT_MONITOR@ are not node names: no target, so the
    canceller follows the default devices, as the recorders would."""
    assert _aec.module_command("@DEFAULT_SOURCE@", "@DEFAULT_MONITOR@") == (
        "load-module libpipewire-module-echo-cancel { "
        "library.name = aec/libspa-aec-webrtc monitor.mode = true "
        'node.name = "simic.echo-cancel.aec" '
        'capture.props = { node.name = "simic.echo-cancel.capture" node.passive = true } '
        'sink.props = { node.name = "simic.echo-cancel.reference" } '
        'source.props = { node.name = "simic.echo-cancel" '
        'node.description = "Simic echo-cancelled mic" priority.session = 0 } '
        "aec.args = { webrtc.gain_control = false webrtc.extended_filter = true } }\n"
    )


def test_named_devices_are_targeted_and_the_reference_is_the_monitors_sink():
    command = _aec.module_command("alsa_input.usb-mic", "alsa_output.speakers.monitor")
    assert (
        'capture.props = { node.name = "simic.echo-cancel.capture" node.passive = true '
        'target.object = "alsa_input.usb-mic" }'
    ) in command
    assert (
        'sink.props = { node.name = "simic.echo-cancel.reference" '
        'target.object = "alsa_output.speakers" }'
    ) in command


def test_start_gives_the_cancelled_source_and_a_process_that_holds_it(tmp_path, fake_pw_cli):
    log = tmp_path / "echo-cancel.log"

    canceller = _aec.start("@DEFAULT_SOURCE@", "@DEFAULT_MONITOR@", log)

    assert canceller.source == "simic.echo-cancel"
    [run] = fake_pw_cli()
    assert run.pid == canceller.pid
    assert run.command == _aec.module_command("@DEFAULT_SOURCE@", "@DEFAULT_MONITOR@")
    assert log.read_text() == "1 = @module:22\n"
    assert process_state(canceller.pid) is not None
    os.kill(canceller.pid, signal.SIGTERM)
    os.waitpid(canceller.pid, 0)


def test_a_module_that_does_not_load_leaves_no_process_and_no_log(tmp_path):
    """The autouse isolate_pipewire's pw-cli: PipeWire without the WebRTC plugin."""
    log = tmp_path / "echo-cancel.log"

    with pytest.raises(_aec.EchoCancelUnavailable) as raised:
        _aec.start("@DEFAULT_SOURCE@", "@DEFAULT_MONITOR@", log)

    assert str(raised.value) == (
        "PipeWire's echo-cancel module did not load (is the WebRTC AEC plugin installed?)"
    )
    assert not log.exists()


def test_without_pw_cli_there_is_no_canceller(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))

    with pytest.raises(_aec.EchoCancelUnavailable) as raised:
        _aec.start("@DEFAULT_SOURCE@", "@DEFAULT_MONITOR@", tmp_path / "echo-cancel.log")

    assert str(raised.value) == "pw-cli not found (PipeWire's command-line tool)"


def test_status_line_says_on_off_or_why_not():
    assert _aec.status_line(True, None) == "Echo cancellation: on"
    assert _aec.status_line(False, None) == "Echo cancellation: off (SIMIC_ECHO_CANCEL)"
    assert _aec.status_line(True, "pw-cli not found") == (
        "Warning: echo cancellation unavailable — pw-cli not found. On speakers, the mic "
        "will also record the other side: use headphones."
    )
