"""The app-menu entry for `simic ui`: a freedesktop .desktop file in the user's
applications directory.

Written by `simic ui --install`, pointing at the simic that wrote it — a venv, a
uv or pipx tool, a system package — rather than at whatever `simic` a desktop
session's PATH happens to find, which is often not ~/.local/bin.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from simic import const

# Characters the Desktop Entry spec requires escaped inside a quoted Exec argument.
_EXEC_RESERVED = re.compile(r'(["`$\\])')


def applications_dir() -> Path:
    """$XDG_DATA_HOME/applications; ~/.local/share when it is unset or relative, as
    the XDG spec requires."""
    base = Path(os.environ.get("XDG_DATA_HOME", ""))
    if not base.is_absolute():
        base = Path.home() / ".local" / "share"
    return base / "applications"


def desktop_entry(executable: Path) -> str:
    quoted = '"' + _EXEC_RESERVED.sub(r"\\\1", str(executable)) + '"'
    # A literal % in Exec would be read as a field code.
    quoted = quoted.replace("%", "%%")
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=simic\n"
        "Comment=Record and transcribe into your notes\n"
        f"Exec={quoted} ui\n"
        "Icon=audio-input-microphone-symbolic\n"
        "Terminal=false\n"
        "Categories=AudioVideo;Audio;Recorder;\n"
        f"StartupWMClass={const.APP_ID}\n"
    )


def install(executable: Path) -> Path:
    """Write the launcher and return its path. Rewrites an existing one: the
    executable it points at may have moved."""
    target = applications_dir() / f"{const.APP_ID}.desktop"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(desktop_entry(executable))
    return target
