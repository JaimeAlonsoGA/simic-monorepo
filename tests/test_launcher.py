"""The applications-menu entry `tapeback ui --install` writes."""

from pathlib import Path

from tapeback import _launcher


def test_the_entry_runs_the_window_of_the_tapeback_that_wrote_it():
    entry = _launcher.desktop_entry(Path("/home/me/.venv/bin/tapeback"))
    assert entry == (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=tapeback\n"
        "Comment=Record and transcribe into your notes\n"
        'Exec="/home/me/.venv/bin/tapeback" ui\n'
        "Icon=audio-input-microphone-symbolic\n"
        "Terminal=false\n"
        "Categories=AudioVideo;Audio;Recorder;\n"
        "StartupWMClass=io.github.jaimealonsoga.tapeback\n"
    )


def test_an_awkward_path_is_escaped_as_the_spec_asks():
    entry = _launcher.desktop_entry(Path('/home/me/My "tools" $HOME 100%/tapeback'))
    assert 'Exec="/home/me/My \\"tools\\" \\$HOME 100%%/tapeback" ui\n' in entry


def test_install_writes_into_xdg_data_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    path = _launcher.install(Path("/opt/tapeback"))

    assert path == tmp_path / "applications" / "io.github.jaimealonsoga.tapeback.desktop"
    assert 'Exec="/opt/tapeback" ui\n' in path.read_text()


def test_a_relative_xdg_data_home_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", "relative")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert _launcher.applications_dir() == tmp_path / ".local" / "share" / "applications"
