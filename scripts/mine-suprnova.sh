#!/usr/bin/env bash
# Mine Qubitcoin (qhash) on Suprnova with CUDA BYOS FP64 (-t 1).
#
# Usage:
#   WALLET=bc1q... ./scripts/mine-suprnova.sh
#   ./scripts/mine-suprnova.sh              # reads .pool-wallet-test.txt
#   ./scripts/mine-suprnova.sh --benchmark  # local bench, no pool
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OFFICIAL="$(cd "$ROOT/../official-miner" && pwd)"
CUDA_HOME="${CUDA_HOME:-$HOME/opt/cuda-13.3}"
BYOS_BUILD="${BYOS_BUILD:-$ROOT/build-byos-cuda}"
POOL="${POOL:-stratum+tcp://qtc.suprnova.cc:5555}"
PASS="${PASS:-x}"

for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy; do
  unset "$v" || true
done

export PATH="$CUDA_HOME/bin:${HOME}/opt/cuda-env/bin:${PATH:-}"
export LD_LIBRARY_PATH="$BYOS_BUILD:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

BIN="$OFFICIAL/qubitcoin-miner"
if [[ ! -x "$BIN" ]]; then
  echo "missing $BIN — run: ./scripts/build-official-byos.sh --cuda" >&2
  exit 1
fi

if [[ "${1:-}" == "--benchmark" ]]; then
  exec "$BIN" -a qhash --benchmark -t 1
fi

WALLET_FILE="$ROOT/.pool-wallet-test.txt"
WALLET="${WALLET:-}"
if [[ -z "$WALLET" && -f "$WALLET_FILE" ]]; then
  WALLET="$(head -1 "$WALLET_FILE")"
fi
if [[ -z "$WALLET" ]]; then
  echo "Set WALLET=bc1q... or create $WALLET_FILE" >&2
  exit 1
fi

USER="${WALLET}.qhashbyos"
echo "pool=$POOL  user=$USER  threads=1 (CUDA)"
exec "$BIN" -a qhash -o "$POOL" -u "$USER" -p "$PASS" -t 1
