#!/usr/bin/env bash
# The desktop pieces of the INSTALLED simic: the tray and the app icon.
#
#     tests/smoke/desktop.sh PYTHON
#
# PYTHON is the interpreter of simic's installed venv. Run by the packaging stands
# after the tray package is installed: its install hook pip-installs the D-Bus
# library, which no other check runs — dbus-next stopped importing on Python 3.15
# with nothing to show it. Building the StatusNotifierItem parses every D-Bus
# signature the tray declares; a container has no session bus to register on.
set -euo pipefail

"$1" - <<'PY'
from simic import const
from simic._sni import StatusNotifierItem
from simic.tray import TrayApp  # noqa: F401 — the whole tray imports

sni = StatusNotifierItem(app_id="simic", title=const.APP_NAME, menu_path="/MenuBar")
assert sni.Title == "Simic", sni.Title
icon = const.ICON_THEME_DIR / "hicolor" / "256x256" / "apps" / f"{const.APP_ID}.png"
assert icon.is_file(), icon
print("desktop: tray and icon in place")
PY
