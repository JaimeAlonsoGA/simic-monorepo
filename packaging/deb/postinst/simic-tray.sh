#!/bin/sh
set -e
/opt/simic/venv/bin/pip install --quiet --upgrade \
    'dbus-fast>=5.2,<6'
# Up to 0.9.10 the tray ran on dbus-next, which no longer imports on Python 3.15.
/opt/simic/venv/bin/pip uninstall --quiet -y dbus-next || true
