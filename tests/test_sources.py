"""The sound server's sources, under the names the window shows."""

from types import SimpleNamespace
from unittest.mock import patch

from simic._sources import Source, list_sources, parse_sources

# `pactl list sources` on PipeWire, cut to what is parsed and what surrounds it.
_LISTING = """Source #58
\tState: IDLE
\tName: alsa_output.pci-0000_03_00.6.analog-stereo.monitor
\tDescription: Monitor of Ryzen HD Audio Controller Analog Stereo
\tDriver: PipeWire
\tProperties:
\t\tdevice.description = "Ryzen HD Audio Controller"
Source #59
\tState: RUNNING
\tName: alsa_input.pci-0000_03_00.6.analog-stereo
\tDescription: Ryzen HD Audio Controller Analog Stereo
"""


def test_parse_names_and_descriptions():
    assert parse_sources(_LISTING) == [
        Source(
            "alsa_output.pci-0000_03_00.6.analog-stereo.monitor",
            "Monitor of Ryzen HD Audio Controller Analog Stereo",
        ),
        Source(
            "alsa_input.pci-0000_03_00.6.analog-stereo", "Ryzen HD Audio Controller Analog Stereo"
        ),
    ]


def test_a_source_without_a_description_is_named_by_its_device():
    listing = "\tName: first\n\tName: second\n\tDescription: \n\tName: last\n"
    assert parse_sources(listing) == [
        Source("first", "first"),
        Source("second", "second"),
        Source("last", "last"),
    ]


def test_a_monitor_is_told_from_a_microphone():
    assert Source("x.monitor", "").is_monitor is True
    assert Source("x.monitor.mic", "").is_monitor is False


def test_list_asks_pactl_in_the_c_locale():
    """pactl translates its own labels: in Spanish `Name:` would not be found."""
    with (
        patch("simic._sources.shutil.which", return_value="/usr/bin/pactl"),
        patch(
            "simic._sources.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout=_LISTING),
        ) as run,
    ):
        sources = list_sources()
    assert [s.name for s in sources] == [
        "alsa_output.pci-0000_03_00.6.analog-stereo.monitor",
        "alsa_input.pci-0000_03_00.6.analog-stereo",
    ]
    assert run.call_args.args[0] == ["pactl", "list", "sources"]
    assert run.call_args.kwargs["env"]["LC_ALL"] == "C"


def test_list_is_empty_without_pactl():
    with patch("simic._sources.shutil.which", return_value=None):
        assert list_sources() == []


def test_list_is_empty_when_no_server_answers():
    with (
        patch("simic._sources.shutil.which", return_value="/usr/bin/pactl"),
        patch(
            "simic._sources.subprocess.run",
            return_value=SimpleNamespace(returncode=1, stdout=""),
        ),
    ):
        assert list_sources() == []
