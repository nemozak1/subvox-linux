#!/usr/bin/env bash
# Pair, trust and connect Apple earbuds from the terminal.
# Put them in pairing mode first (case open, hold the system button until the light pulses white).
#   tools/pair.sh                 # looks for a device whose name matches Beats|AirPods
#   tools/pair.sh "Powerbeats"    # custom name pattern
#   tools/pair.sh AA:BB:CC:DD:EE:FF
set -euo pipefail

pattern="${1:-Beats|AirPods}"
mac=""
if [[ "$pattern" =~ ^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$ ]]; then
  mac="$pattern"
else
  echo "scanning up to 30s for a device matching /$pattern/ ..."
  for _ in $(seq 1 6); do
    bluetoothctl --timeout 5 scan on >/dev/null 2>&1 || true
    mac=$(bluetoothctl devices | grep -iE "$pattern" | head -1 | awk '{print $2}')
    [[ -n "$mac" ]] && break
  done
fi
if [[ -z "$mac" ]]; then
  echo "not found. Are the buds in pairing mode (white pulsing light)?" >&2
  exit 1
fi

name=$(bluetoothctl devices | awk -v m="$mac" '$2==m {$1="";$2=""; sub(/^  /,""); print}')
echo "found: $mac  $name"
bluetoothctl pair "$mac"
bluetoothctl trust "$mac"
bluetoothctl connect "$mac"
sleep 2
bluetoothctl info "$mac" | grep -E 'Name|Paired|Trusted|Connected|UUID: (Audio Sink|Handsfree|Vendor)'
