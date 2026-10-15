"""The applications-menu entry `simic ui --install` writes."""

from pathlib import Path

from simic import _launcher


def test_the_entry_runs_the_window_of_the_simic_that_wrote_it():
    entry = _launcher.desktop_entry(Path("/home/me/.venv/bin/simic"))
    assert entry == (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Simic\n"
        "Comment=Record and transcribe into your notes\n"
        'Exec="/home/me/.venv/bin/simic" ui\n'
        "Icon=io.github.jaimealonsoga.simic\n"
        "Terminal=false\n"
        "Categories=AudioVideo;Audio;Recorder;\n"
        "StartupWMClass=io.github.jaimealonsoga.simic\n"
    )


def test_an_awkward_path_is_escaped_as_the_spec_asks():
    entry = _launcher.desktop_entry(Path('/home/me/My "tools" $HOME 100%/simic'))
    assert 'Exec="/home/me/My \\"tools\\" \\$HOME 100%%/simic" ui\n' in entry


def test_install_writes_into_xdg_data_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    path = _launcher.install(Path("/opt/simic"))

    assert path == tmp_path / "applications" / "io.github.jaimealonsoga.simic.desktop"
    assert 'Exec="/opt/simic" ui\n' in path.read_text()


def test_install_puts_the_icon_where_the_entry_names_it(tmp_path, monkeypatch):
    """`Icon=io.github.jaimealonsoga.simic` is a name the desktop resolves in the
    user's hicolor theme, so every size has to land there."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    _launcher.install(Path("/opt/simic"))

    hicolor = tmp_path / "icons" / "hicolor"
    assert sorted(p.relative_to(hicolor).as_posix() for p in hicolor.rglob("*.png")) == [
        "128x128/apps/io.github.jaimealonsoga.simic.png",
        "256x256/apps/io.github.jaimealonsoga.simic.png",
        "48x48/apps/io.github.jaimealonsoga.simic.png",
        "512x512/apps/io.github.jaimealonsoga.simic.png",
    ]


def test_install_replaces_an_older_icon(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    old = tmp_path / "icons" / "hicolor" / "48x48" / "apps" / "io.github.jaimealonsoga.simic.png"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old icon")

    _launcher.install(Path("/opt/simic"))

    assert old.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_relative_xdg_data_home_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", "relative")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert _launcher.applications_dir() == tmp_path / ".local" / "share" / "applications"
