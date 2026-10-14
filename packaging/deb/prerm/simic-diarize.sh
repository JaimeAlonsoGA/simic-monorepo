#!/bin/sh
set -e
# Only uninstall on full removal, not on upgrade.
if [ "$1" = "remove" ] || [ "$1" = "purge" ]; then
    /opt/simic/venv/bin/pip uninstall --quiet -y \
        pyannote-audio torch torchaudio || true
fi
