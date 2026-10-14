#!/bin/sh
set -e
if [ "$1" = "remove" ] || [ "$1" = "purge" ]; then
    /opt/simic/venv/bin/pip uninstall --quiet -y dbus-next pystray Pillow || true
fi
