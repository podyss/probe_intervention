#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COLLECT_SCRIPT="$SCRIPT_DIR/collect_deep_r1.sh"
THRESHOLD_MB=${THRESHOLD_MB:-75000}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-600}

if ! [[ "$THRESHOLD_MB" =~ ^[0-9]+$ ]] || ! [[ "$INTERVAL_SECONDS" =~ ^[0-9]+$ ]] || (( INTERVAL_SECONDS == 0 )); then
  echo "THRESHOLD_MB must be a non-negative integer and INTERVAL_SECONDS must be a positive integer." >&2
  exit 2
fi

if [[ ! -f "$COLLECT_SCRIPT" ]]; then
  echo "Collection script not found: $COLLECT_SCRIPT" >&2
  exit 2
fi

echo "Monitoring GPU pairs 0,1 and 2,3; requiring ${THRESHOLD_MB} MiB free per GPU."

while true; do
  gpu_memory=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null) || gpu_memory=""

  pair_is_available() {
    local first="$1" second="$2" gpu free_mb
    for gpu in "$first" "$second"; do
      free_mb=$(awk -F, -v target="$gpu" '
        {
          gpu_index = $1
          free = $2
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", gpu_index)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", free)
          if (gpu_index == target && free ~ /^[0-9]+$/) {
            print free
            found = 1
            exit
          }
        }
        END { if (!found) exit 1 }
      ' <<< "$gpu_memory") || free_mb=""

      if [[ -z "$free_mb" ]] || (( free_mb < THRESHOLD_MB )); then
        echo "GPU $gpu unavailable or below threshold (${free_mb:-unknown} MiB free)."
        return 1
      fi
      echo "GPU $gpu available ($free_mb MiB free)."
    done
    return 0
  }

  if [[ -n "$gpu_memory" ]] && pair_is_available 0 1; then
    devices="0,1"
  elif [[ -n "$gpu_memory" ]] && pair_is_available 2 3; then
    devices="2,3"
  else
    devices=""
  fi

  if [[ -n "$devices" ]]; then
    echo "GPUs $devices meet the memory requirement. Starting collection."
    CUDA_DEVICES="$devices" bash "$COLLECT_SCRIPT"
    exit $?
  fi

  echo "No eligible GPU pair; retrying in ${INTERVAL_SECONDS} seconds."
  sleep "$INTERVAL_SECONDS"
done
