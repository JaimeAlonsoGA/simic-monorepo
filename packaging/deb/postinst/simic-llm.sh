#!/bin/sh
set -e
/opt/simic/venv/bin/pip install --quiet --upgrade \
    'anthropic>=0.40.0,<1' \
    'openai>=1.50.0,<3'
