#!/usr/bin/env bash
set -euo pipefail
mkdir -p build
p4c-bm2-ss --p4v 16 --p4runtime-files build/aurora.p4info.txt -o build/aurora.json p4/aurora_telemetry.p4
echo 'BMv2 JSON and P4Info generated in build/'
