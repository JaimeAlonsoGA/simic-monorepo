"""Regression: start must not crash when peer `simic stop` already finished."""

from unittest.mock import MagicMock, patch

from simic.cli import cli
from simic.recorder import NoActiveRecording
from simic.settings import Settings


def test_start_exits_cleanly_when_peer_stop_finished(runner, vault_env):
    """If `simic stop` already cleared the session, start must not crash."""
    recorder = MagicMock()
    recorder.start.return_value = "peer-stopped"
    # Wait loop sees active once, then peer stop claimed the session.
    recorder.is_recording.side_effect = [True, False]
    recorder.has_session.return_value = False

    with (
        patch("simic.cli.get_settings", return_value=Settings(vault_path=vault_env)),
        patch("simic.cli.Recorder", return_value=recorder),
        patch("simic.cli.detect_devices", return_value=("mon", "mic")),
        patch("simic.pipeline.process_session") as mock_process,
        patch("time.sleep"),
    ):
        result = runner.invoke(cli, ["start", "peer-stopped", "--no-summarize"])

    assert result.exit_code == 0, result.output + str(result.exception or "")
    recorder.stop.assert_not_called()
    mock_process.assert_not_called()


def test_start_exits_cleanly_when_stop_races_after_is_recording(runner, vault_env):
    """The session is still there after the loop; the stop loses the claim to a peer
    stop and raises NoActiveRecording → exit 0, and nothing is processed."""
    recorder = MagicMock()
    recorder.start.return_value = "race-window"
    # The loop ends; the session still looks unclaimed, then the peer stop claims it.
    recorder.is_recording.side_effect = [True, False]
    recorder.has_session.return_value = True
    recorder.stopped_channels.return_value = []
    recorder.stop.side_effect = NoActiveRecording("No recording in progress.")

    with (
        patch("simic.cli.get_settings", return_value=Settings(vault_path=vault_env)),
        patch("simic.cli.Recorder", return_value=recorder),
        patch("simic.cli.detect_devices", return_value=("mon", "mic")),
        patch("simic.pipeline.process_session") as mock_process,
        patch("time.sleep"),
    ):
        result = runner.invoke(cli, ["start", "race-window", "--no-summarize"])

    assert result.exit_code == 0, result.output + str(result.exception or "")
    assert result.exception is None
    recorder.stop.assert_called_once()
    mock_process.assert_not_called()
