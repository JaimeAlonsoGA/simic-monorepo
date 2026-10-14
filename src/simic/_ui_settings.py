"""The settings page of `simic ui`: `_config.SECTIONS`, drawn with libadwaita.

Every change is saved the moment it is made — there is no Save button to forget. A
change the settings would refuse is not written; the row returns to the value in
effect and a toast says why. Each section carries the verdict of the `doctor` checks
behind it, so whether this part of a recording will work is visible where it is set.
"""

from __future__ import annotations

import os
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
from gi.repository import Adw, Gio, GLib, Gtk  # noqa: E402  # ty: ignore[unresolved-import]
from pydantic import ValidationError  # noqa: E402

from simic import _config  # noqa: E402
from simic._config import Choice, Field, Section  # noqa: E402
from simic._doctor import Check  # noqa: E402
from simic._sources import Source, list_sources  # noqa: E402
from simic.settings import USER_ENV_FILE, Settings, get_settings  # noqa: E402

_OUTCOME_ICONS = {
    "ok": ("object-select-symbolic", "success"),
    "warn": ("dialog-warning-symbolic", "warning"),
    "fail": ("dialog-error-symbolic", "error"),
}


class SettingsPage(Adw.PreferencesPage):
    def __init__(
        self,
        settings: Settings,
        on_saved: Callable[[Settings], None],
        on_error: Callable[[str], None],
    ) -> None:
        super().__init__()
        self._values = _config.current_values(settings)
        self._on_saved = on_saved
        self._on_error = on_error
        self._locked = _config.locked(os.environ)
        self._sources: list[Source] = list_sources()
        self._rows: dict[str, Adw.PreferencesRow] = {}
        self._choices: dict[str, list[Choice]] = {}
        self._groups: dict[str, tuple[Adw.PreferencesGroup, Gtk.Image]] = {}
        # Set while rows are brought in line with the values, so their signals are not
        # taken for the user's edits.
        self._syncing = False

        for section in _config.SECTIONS:
            self.add(self._group(section))
        self._sync()

    # --- building -------------------------------------------------------------

    def _group(self, section: Section) -> Adw.PreferencesGroup:
        group = Adw.PreferencesGroup(title=section.title)
        mark = Gtk.Image(visible=False, valign=Gtk.Align.CENTER)
        group.set_header_suffix(mark)
        self._groups[section.title] = (group, mark)
        for f in section.fields:
            row = self._row(f)
            if f.name in self._locked:
                row.set_sensitive(False)
                row.set_tooltip_text(f"Set by {_config.env_key(f.name)} in the environment")
            self._rows[f.name] = row
            group.add(row)
        return group

    def _row(self, f: Field) -> Adw.PreferencesRow:
        if f.kind == "switch":
            row = Adw.SwitchRow(title=f.title)
            row.connect("notify::active", self._switched, f.name)
            return row
        if f.kind == "choice":
            row = Adw.ComboRow(title=f.title)
            row.connect("notify::selected", self._chosen, f.name)
            return row
        if f.kind == "folder":
            row = Adw.ActionRow(title=f.title, subtitle_selectable=True)
            button = Gtk.Button(
                icon_name="folder-open-symbolic",
                valign=Gtk.Align.CENTER,
                css_classes=["flat"],
                tooltip_text="Choose a folder",
            )
            button.connect("clicked", self._choose_folder, f.name)
            row.add_suffix(button)
            row.set_activatable_widget(button)
            return row
        if f.kind == "text":
            row = Adw.EntryRow(title=f.title, show_apply_button=True)
        else:
            row = Adw.PasswordEntryRow(title=f.title, show_apply_button=True)
        row.connect("apply", self._applied, f.name)
        return row

    # --- the values, shown ------------------------------------------------------

    def _sync(self) -> None:
        """Bring every row in line with the values in effect."""
        self._syncing = True
        try:
            for name, row in self._rows.items():
                f = _config.FIELDS[name]
                value = self._values[name]
                row.set_visible(f.visible(self._values))
                if isinstance(row, Adw.SwitchRow):
                    row.set_active(value == "true")
                elif isinstance(row, Adw.ComboRow):
                    self._sync_choices(row, f, value)
                elif isinstance(row, Adw.EntryRow):
                    if row.get_text() != value:
                        row.set_text(value)
                elif isinstance(row, Adw.ActionRow):
                    row.set_subtitle(value)
        finally:
            self._syncing = False

    def _sync_choices(self, row: Adw.ComboRow, f: Field, value: str) -> None:
        choices = _config.choices_for(f, self._values, self._sources)
        if choices != self._choices.get(f.name):
            self._choices[f.name] = choices
            row.set_model(Gtk.StringList.new([choice.label for choice in choices]))
        index = next((i for i, c in enumerate(choices) if c.value == value), 0)
        row.set_selected(index)

    def show_health(self, checks: list[Check]) -> None:
        """Mark each section with the worst of its checks, and say what is wrong."""
        by_name = {check.name: check for check in checks}
        for section in _config.SECTIONS:
            group, mark = self._groups[section.title]
            found = [by_name[name] for name in section.checks if name in by_name]
            if not found:
                continue
            worst = max(found, key=lambda c: list(_OUTCOME_ICONS).index(c.outcome))
            icon, style = _OUTCOME_ICONS[worst.outcome]
            mark.set_from_icon_name(icon)
            mark.set_css_classes([style])
            mark.set_visible(True)
            problems = [check.detail for check in found if check.outcome != "ok"]
            group.set_description("\n".join(problems) or None)

    # --- edits ------------------------------------------------------------------

    def _switched(self, row: Adw.SwitchRow, _param: object, name: str) -> None:
        if not self._syncing:
            self._commit(name, "true" if row.get_active() else "false")

    def _chosen(self, row: Adw.ComboRow, _param: object, name: str) -> None:
        choices = self._choices.get(name, [])
        index = row.get_selected()
        if not self._syncing and index < len(choices):
            self._commit(name, choices[index].value)

    def _applied(self, row: Adw.EntryRow, name: str) -> None:
        self._commit(name, row.get_text().strip())

    def _choose_folder(self, button: Gtk.Button, name: str) -> None:
        dialog = Gtk.FileDialog(title="Choose a folder", modal=True)
        current = Path(self._values[name])
        if current.is_dir():
            dialog.set_initial_folder(Gio.File.new_for_path(str(current)))
        dialog.select_folder(button.get_root(), None, self._folder_chosen, name)

    def _folder_chosen(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult, name: str) -> None:
        try:
            folder = dialog.select_folder_finish(result)
        except GLib.Error:
            return  # dismissed
        path = folder.get_path() if folder is not None else None
        if path:
            self._commit(name, path)

    def _commit(self, name: str, value: str) -> None:
        if value == self._values[name]:
            return
        try:
            _config.save(USER_ENV_FILE, _config.changes_for(name, value))
        except ValidationError as exc:
            self._on_error(_config.describe(exc))
            self._sync()
            return
        except OSError as exc:
            self._on_error(f"Could not save to {USER_ENV_FILE}: {exc.strerror}")
            self._sync()
            return
        settings = get_settings()
        self._values = _config.current_values(settings)
        self._sync()
        self._on_saved(settings)
