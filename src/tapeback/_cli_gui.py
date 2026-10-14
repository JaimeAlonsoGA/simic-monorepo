"""The graphical commands, `tapeback ui` and `tapeback tray`, apart from cli.py.

Both import their toolkit only when run: GTK and dbus-next are optional, and every
other command must start without them.
"""

import shutil
from pathlib import Path

import click

from tapeback import _launcher

_GTK_MISSING = (
    "The window needs GTK 4 and libadwaita 1.4+ for Python (PyGObject).\n"
    "  Debian/Ubuntu: sudo apt install python3-gi gir1.2-adw-1\n"
    "  Fedora:        sudo dnf install python3-gobject libadwaita\n"
    "  Arch:          sudo pacman -S python-gobject libadwaita\n"
    "PyGObject comes with the system Python: install tapeback where it can see it —\n"
    "a venv made with --system-site-packages, or pipx install --system-site-packages."
)


@click.command()
@click.option(
    "--install",
    is_flag=True,
    help="Add tapeback to the applications menu, then exit",
)
def ui(install: bool) -> None:
    """Open the tapeback window: record, follow transcriptions, change settings.

    Settings changed there are saved to ~/.config/tapeback/.env, the file every
    command reads. A recording or transcription keeps going when the window closes.
    """
    if install:
        executable = shutil.which("tapeback")
        if executable is None:
            raise click.ClickException("tapeback is not on PATH: the launcher needs to find it.")
        path = _launcher.install(Path(executable))
        click.echo(f"Added to the applications menu: {path}", err=True)
        return
    try:
        from tapeback.ui import run_ui
    except (ImportError, ValueError):
        # ValueError: gi is there, but not the GTK 4 or libadwaita typelib asked for.
        raise click.ClickException(_GTK_MISSING) from None
    raise SystemExit(run_ui())


@click.command()
def tray() -> None:
    """Run tapeback as a system tray icon.

    Provides a graphical interface for start/stop recording
    without needing a terminal. Right-click the icon for the menu.

    \b
    Requires the tray extra:
      uv pip install tapeback[tray]
    """
    try:
        from tapeback.tray import run_tray
    except ImportError:
        raise click.ClickException(
            "System tray requires pystray and Pillow. Install with: uv pip install tapeback[tray]"
        ) from None

    run_tray()
