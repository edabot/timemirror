#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== TimeMirror release build ==="
echo "Started: $(date)"
echo

make notarize

echo
echo "Done: $(date)"
open -R TimeMirror_notarized.zip
