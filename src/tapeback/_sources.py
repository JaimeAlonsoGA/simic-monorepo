"""The audio sources the sound server offers, under the names a person recognises.

`pactl list sources short` gives only the device names (alsa_input.pci-0000_03_00.6...),
and `pactl -f json` reports a null description on PipeWire, so the long listing is
parsed for its `Name:` and `Description:` lines. Under the C locale: pactl translates
its own labels, while the description comes from the sound server as it is.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

_MONITOR_SUFFIX = ".monitor"
_NAME = "Name:"
_DESCRIPTION = "Description:"


@dataclass(frozen=True)
class Source:
    name: str
    description: str

    @property
    def is_monitor(self) -> bool:
        """What the speakers play, rather than a microphone."""
        return self.name.endswith(_MONITOR_SUFFIX)


def parse_sources(listing: str) -> list[Source]:
    """Sources in the order pactl lists them. One without a description is named by
    its device name."""
    sources: list[Source] = []
    name = ""
    for raw in listing.splitlines():
        line = raw.strip()
        if line.startswith(_NAME):
            if name:
                sources.append(Source(name, name))
            name = line.removeprefix(_NAME).strip()
        elif line.startswith(_DESCRIPTION) and name:
            sources.append(Source(name, line.removeprefix(_DESCRIPTION).strip() or name))
            name = ""
    if name:
        sources.append(Source(name, name))
    return sources


def list_sources() -> list[Source]:
    """Every source the sound server has now. Empty when there is no pactl or no
    server answers — the caller shows only what is configured, and `doctor` says why."""
    if not shutil.which("pactl"):
        return []
    result = subprocess.run(
        ["pactl", "list", "sources"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "LC_ALL": "C"},
    )
    if result.returncode != 0:
        return []
    return parse_sources(result.stdout)
