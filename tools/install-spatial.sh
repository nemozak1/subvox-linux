#!/usr/bin/env bash
# Install the binaural sink into the user PipeWire session and make it the default output.
#   tools/install-spatial.sh            # install + restart filter-chain service
#   tools/install-spatial.sh --remove
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
dst=~/.config/pipewire/filter-chain.conf.d/60-subvox-spatial.conf
if [[ "${1:-}" == "--remove" ]]; then
  rm -f "$dst"; systemctl --user restart filter-chain; echo removed; exit 0
fi
mkdir -p "$(dirname "$dst")"
cp "$here/spatial/subvox-spatial.conf" "$dst"
systemctl --user restart filter-chain
for _ in $(seq 1 20); do
  id=$(pw-dump 2>/dev/null | python3 -I -c 'import json,sys
for o in json.load(sys.stdin):
    if o.get("info",{}).get("props",{}).get("node.name")=="subvox_spatial": print(o["id"])' | head -1)
  [[ -n "$id" ]] && break; sleep 0.5
done
[[ -n "${id:-}" ]] || { echo "sink did not appear; journalctl --user -u filter-chain" >&2; exit 1; }
wpctl set-default "$id"
echo "default sink -> $id Spatial (Powerbeats)"
