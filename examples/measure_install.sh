#!/usr/bin/env bash
# Measures what each embeddings integration installs: a fresh virtual
# environment per option, then the size of its site-packages.
#
#   examples/measure_install.sh
set -euo pipefail
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

measure() {
    local name="$1"; shift
    python3 -m venv "$work/$name"
    "$work/$name/bin/pip" install -q --upgrade pip
    "$work/$name/bin/pip" install -q "$@"
    local site; site="$("$work/$name/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
    printf '%-60s %8s MB  %4s packages\n' "$name" \
        "$(du -sm "$site" | cut -f1)" \
        "$("$work/$name/bin/pip" list 2>/dev/null | tail -n +3 | wc -l)"
    rm -rf "$work/$name"
}

measure "langchain-kjarni" langchain-kjarni
measure "langchain-huggingface + sentence-transformers" langchain-huggingface sentence-transformers
measure "langchain-huggingface + sentence-transformers (CPU PyTorch)" \
    --extra-index-url https://download.pytorch.org/whl/cpu torch langchain-huggingface sentence-transformers
