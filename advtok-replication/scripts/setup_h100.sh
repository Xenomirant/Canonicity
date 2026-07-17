#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3.10}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

cd "${REPO_DIR}"
"${PYTHON_BOOTSTRAP}" -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install --index-url "${CUDA_INDEX_URL}" torch==2.6.0
.venv/bin/python -m pip install -r requirements-h100.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python scripts/bootstrap_autodan.py
.venv/bin/python -m nltk.downloader -d .nltk_data punkt punkt_tab stopwords wordnet omw-1.4
PYTHON_BIN=.venv/bin/python scripts/fetch_data.sh

echo "Setup complete. Export HF_TOKEN (and OPENAI_API_KEY for rubric evaluation) before running."
