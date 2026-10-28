"""`simic ui`: record, follow the recording to its note, and set simic up.

Two pages. Record: one button, the time, and what happens to each recording after it —
being transcribed and how far, waiting to be, the notes already written, which open in
Obsidian. Settings: `_ui_settings`.

Not everything recorded is a meeting — a voice memo, a lecture — so the window calls
them transcripts.

The window is a view over the CLI's own state (`_activity`), refreshed every second,
so it agrees with `simic status` and can be closed at any moment: recording and
transcription run in processes of their own. Anything slow — starting, stopping,
the `doctor` checks — runs off the main loop, and the page shows it is under way.
"""

from __future__ import annotations

import datetime
import threading
from collections.abc import Callable
from pathlib import Path

# PyGObject comes with the system Python (see _cli_gui._GTK_MISSING), never from the
# lockfile, so a checker running in the project venv cannot see it.
import gi  # ty: ignore[unresolved-import]

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
# PyGObject resolves the versions above when the namespace is first imported, so they
# have to be pinned before this import, as its documentation requires. Its members are
# generated from the GObject typelibs at runtime: no static checker can see them.
from gi.repository import Adw, Gdk, Gio, GLib, Gtk  # noqa: E402  # ty: ignore[unresolved-import]

from simic import _activity, const  # noqa: E402
from simic._activity import Activity, Note  # noqa: E402
from simic._doctor import Check, run_checks  # noqa: E402
from simic._ui_settings import SettingsPage  # noqa: E402
from simic.recorder import Recorder  # noqa: E402
from simic.settings import Settings, get_settings  # noqa: E402
from simic.vault import obsidian_uri  # noqa: E402

REFRESH_SECONDS = 1
# Notes listed at first, and added by each "Show more".
NOTES_PAGE = 5
# Opens square, the record page whole; a longer list scrolls.
WINDOW_SIZE = 540
_SECONDS_PER_HOUR = 3600
_SECONDS_PER_MINUTE = 60
_OBSIDIAN_SCHEME = "obsidian"

_CSS = """
.record-button {
  min-width: 132px;
  min-height: 132px;
  border-radius: 9999px;
}
.record-button image { -gtk-icon-size: 44px; }
.clock {
  font-size: 56px;
  font-weight: 300;
  font-feature-settings: "tnum";
}
"""


def _run_off_main_loop[T](
    work: Callable[[], T],
    done: Callable[[T], None],
    failed: Callable[[Exception], None],
) -> None:
    """Run `work` in a thread and hand its result, or its error, back to the main loop:
    GTK may only be touched from there. Not a daemon: a window closed while a recording
    stops waits for the stop to finish, rather than leaving its recorders running."""

    def target() -> None:
        try:
            result = work()
        except Exception as exc:
            GLib.idle_add(failed, exc)
            return
        GLib.idle_add(done, result)

    threading.Thread(target=target).start()


def _clock(started_at: datetime.datetime) -> str:
    seconds = max(0, int((datetime.datetime.now(datetime.UTC) - started_at).total_seconds()))
    hours, rest = divmod(seconds, _SECONDS_PER_HOUR)
    minutes, seconds = divmod(rest, _SECONDS_PER_MINUTE)
    return f"{hours}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"


class RecordPage(Gtk.ScrolledWindow):
    def __init__(self, window: Window) -> None:
        super().__init__(hscrollbar_policy=Gtk.PolicyType.NEVER)
        self._window = window
        self._activity: Activity | None = None
        self._busy = False
        # How many notes the list asks for; "Show more" raises it.
        self.note_limit = NOTES_PAGE

        box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=28,
            margin_top=32,
            margin_bottom=32,
            margin_start=24,
            margin_end=24,
        )
        self.set_child(Adw.Clamp(maximum_size=460, child=box))

        self._name = Adw.EntryRow(title="Name")
        names = Gtk.ListBox(css_classes=["boxed-list"], selection_mode=Gtk.SelectionMode.NONE)
        names.append(self._name)
        box.append(names)

        self._time = Gtk.Label(label="00:00", css_classes=["clock"])
        self._state = Gtk.Label(css_classes=["dim-label"])
        heading = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        heading.append(self._time)
        heading.append(self._state)
        box.append(heading)

        self._button = Gtk.Button(halign=Gtk.Align.CENTER, css_classes=["record-button"])
        self._button.connect("clicked", self._clicked)
        box.append(self._button)

        self._jobs = Adw.PreferencesGroup(title="Transcripts", margin_top=12)
        self._job_rows: list[Gtk.Widget] = []
        box.append(self._jobs)

    # --- refresh ------------------------------------------------------------------

    def refresh(self, activity: Activity) -> None:
        recording = activity.recording
        was_recording = self._activity is not None and self._activity.recording is not None
        # The name stays where it is while recording, so nothing on the page moves.
        self._name.set_sensitive(recording is None and not self._busy)
        if recording is None:
            if was_recording:
                self._name.set_text("")
            self._time.set_label("00:00")
            self._time.remove_css_class("error")
            self._show_state("Ready", warning=False)
            self._show_button("media-record-symbolic", "Record", "destructive-action")
        else:
            if self._name.get_text() != recording.name:
                self._name.set_text(recording.name)
            self._time.set_label(_clock(recording.started_at))
            self._time.add_css_class("error")
            if recording.stalled:
                self._show_state("A recorder stopped", warning=True)
            else:
                self._show_state("Recording", warning=False)
            self._show_button("media-playback-stop-symbolic", "Stop", "suggested-action")
        self._button.set_sensitive(not self._busy)
        if activity != self._activity:
            self._activity = activity
            self._show_jobs(activity)

    def _show_state(self, text: str, *, warning: bool) -> None:
        if self._busy:
            return  # "Starting…" or "Stopping…" until the work is done
        self._state.set_label(text)
        self._state.set_css_classes(["warning"] if warning else ["dim-label"])

    def _show_button(self, icon: str, tooltip: str, style: str) -> None:
        self._button.set_icon_name(icon)
        self._button.set_tooltip_text(tooltip)
        self._button.set_css_classes(["record-button", "circular", style])

    def _show_jobs(self, activity: Activity) -> None:
        for row in self._job_rows:
            self._jobs.remove(row)
        rows: list[Gtk.Widget] = []
        for job in activity.transcribing:
            row = Adw.ActionRow(title=job.name, subtitle=job.status or "Transcribing")
            row.add_prefix(Gtk.Spinner(spinning=True))
            rows.append(row)
        for name in activity.pending:
            row = Adw.ActionRow(title=name, subtitle="Not transcribed")
            button = Gtk.Button(label="Transcribe", valign=Gtk.Align.CENTER, css_classes=["pill"])
            button.connect("clicked", self._recover, name)
            row.add_suffix(button)
            rows.append(row)
        now = datetime.datetime.now().astimezone()
        rows.extend(self._note_row(note, now) for note in activity.notes)
        if activity.more_notes:
            more = Gtk.Button(label="Show more", css_classes=["flat"], margin_top=6)
            more.connect("clicked", self._show_more)
            rows.append(more)
        for row in rows:
            self._jobs.add(row)
        self._job_rows = rows
        self._jobs.set_visible(bool(rows))

    def _note_row(self, note: Note, now: datetime.datetime) -> Adw.ActionRow:
        row = Adw.ActionRow(
            title=GLib.markup_escape_text(note.path.stem),
            subtitle=_activity.when(note.written_at, now),
            activatable=True,
        )
        row.add_prefix(Gtk.Image(icon_name="object-select-symbolic", css_classes=["success"]))
        row.add_suffix(Gtk.Image(icon_name="go-next-symbolic"))
        row.connect("activated", lambda _row: self._window.open_note(note.path))
        return row

    def _show_more(self, _button: Gtk.Button) -> None:
        self.note_limit += NOTES_PAGE
        self._window.refresh()

    # --- actions ------------------------------------------------------------------

    def _clicked(self, _button: Gtk.Button) -> None:
        recording = self._activity is not None and self._activity.recording is not None
        recorder = Recorder()
        settings = self._window.settings
        name = self._name.get_text().strip()

        def work() -> str:
            if recording:
                return _activity.stop(recorder)
            return _activity.start(settings, recorder, name)

        self._state.set_label("Stopping…" if recording else "Starting…")
        self._state.set_css_classes(["dim-label"])
        self._busy = True
        self._button.set_sensitive(False)
        self._name.set_sensitive(False)
        _run_off_main_loop(work, self._finished, self._failed)

    def _finished(self, _name: str) -> None:
        self._busy = False
        self._window.refresh()

    def _failed(self, exc: Exception) -> None:
        self._busy = False
        self._window.toast(str(exc))
        self._window.refresh()

    def _recover(self, button: Gtk.Button, name: str) -> None:
        button.set_sensitive(False)
        settings = self._window.settings
        _run_off_main_loop(
            lambda: _activity.recover(settings, Recorder(), name),
            lambda _none: self._window.refresh(),
            self._failed,
        )


class Window(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application) -> None:
        super().__init__(
            application=app,
            title=const.APP_NAME,
            default_width=WINDOW_SIZE,
            default_height=WINDOW_SIZE,
        )
        self.settings = get_settings()
        self._recorder = Recorder()
        # The banner's two sources: an unusable recordings folder, a failing check.
        self._unreadable = ""
        self._failing = ""

        self._toasts = Adw.ToastOverlay()
        self._banner = Adw.Banner()
        stack = Adw.ViewStack()
        self._record = RecordPage(self)
        self._settings_page = SettingsPage(self.settings, self._saved, self.toast)
        stack.add_titled_with_icon(self._record, "record", "Record", "media-record-symbolic")
        stack.add_titled_with_icon(
            self._settings_page, "settings", "Settings", "emblem-system-symbolic"
        )

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        content.append(self._banner)
        content.append(stack)
        stack.set_vexpand(True)
        self._toasts.set_child(content)

        header = Adw.HeaderBar(
            title_widget=Adw.ViewSwitcher(stack=stack, policy=Adw.ViewSwitcherPolicy.WIDE)
        )
        view = Adw.ToolbarView(content=self._toasts)
        view.add_top_bar(header)
        self.set_content(view)

        self.refresh()
        GLib.timeout_add_seconds(REFRESH_SECONDS, self._tick)
        self.check_health()

    def _tick(self) -> bool:
        self.refresh()
        return GLib.SOURCE_CONTINUE

    def refresh(self) -> None:
        try:
            activity = _activity.read_activity(
                self.settings, self._recorder, self._record.note_limit
            )
        except (OSError, RuntimeError) as exc:
            # The sessions directory is unusable (see recorder.session_root): say so in
            # the banner rather than every second in a toast.
            self._unreadable = str(exc)
        else:
            self._unreadable = ""
            self._record.refresh(activity)
        self._show_banner()

    def _show_banner(self) -> None:
        """What stops a recording now: the recordings folder, else a failing check."""
        message = self._unreadable or self._failing
        if message != self._banner.get_title():
            self._banner.set_title(message)
        self._banner.set_revealed(bool(message))

    def check_health(self) -> None:
        settings = self.settings
        _run_off_main_loop(lambda: run_checks(settings), self._show_health, self._failed_checks)

    def _show_health(self, checks: list[Check]) -> None:
        self._settings_page.show_health(checks)
        failing = [check for check in checks if check.outcome == "fail"]
        self._failing = f"{failing[0].name}: {failing[0].detail}" if failing else ""
        self._show_banner()

    def _failed_checks(self, exc: Exception) -> None:
        self.toast(f"Could not check the setup: {exc}")

    def _saved(self, settings: Settings) -> None:
        self.settings = settings
        self.check_health()

    def toast(self, message: str) -> None:
        self._toasts.add_toast(Adw.Toast(title=GLib.markup_escape_text(message), timeout=6))

    def open_note(self, path: Path) -> None:
        """In Obsidian when it is installed: the default app for .md is often an editor."""
        if Gio.AppInfo.get_default_for_uri_scheme(_OBSIDIAN_SCHEME) is not None:
            Gtk.UriLauncher(uri=obsidian_uri(path)).launch(self, None, None)
        else:
            Gtk.FileLauncher(file=Gio.File.new_for_path(str(path))).launch(self, None, None)


class App(Adw.Application):
    def __init__(self) -> None:
        super().__init__(application_id=const.APP_ID)
        self.connect("activate", self._activate)

    def _activate(self, _app: Adw.Application) -> None:
        window = self.get_active_window()
        if window is None:
            css = Gtk.CssProvider()
            css.load_from_string(_CSS)
            display = Gdk.Display.get_default()
            if display is not None:
                Gtk.StyleContext.add_provider_for_display(
                    display, css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
                )
                # The icon ships in the package, so the window has it before
                # `simic ui --install` puts a copy in the user's theme.
                Gtk.IconTheme.get_for_display(display).add_search_path(str(const.ICON_THEME_DIR))
            Gtk.Window.set_default_icon_name(const.APP_ID)
            window = Window(self)
        window.present()


def run_ui() -> int:
    return App().run([])
