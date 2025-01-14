#!/usr/bin/env bash
set -eu

INPUT="vis.json"
OUTPUT="vis.md"

userexit() {
    printf "\e[1;31merror\e[0m: $1\n" "${@:2}"
    exit 1
}

usersuccess() {
    printf "\e[1;32msuccess\e[0m: $1\n" "${@:2}"
}

if [[ -f "$OUTPUT" && -f ".$OUTPUT.sha256" ]]; then
    sha256sum --check --status --ignore-missing ".$OUTPUT.sha256" || userexit "$OUTPUT has been modified since last generated"
fi

curl https://raw.githubusercontent.com/Riernar/arm5/refs/heads/main/vis/vis.py | uv run --script - -i "$INPUT" -o "$OUTPUT" "$@"
sha256sum "$OUTPUT" >".$OUTPUT.sha256"
usersuccess "generated $OUTPUT and a sha256 checksum in .$OUTPUT.sha256"
