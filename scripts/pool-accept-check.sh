#!/usr/bin/env bash
# Phase 5.3 — pool share accept-rate check (FP64 CUDA BYOS, -t 1).
#
# Needs real GPU + outbound stratum TCP (not via HTTP CONNECT proxy).
# Usage:
#   ./scripts/pool-accept-check.sh                 # Suprnova (default), ~10 min
#   ./scripts/pool-accept-check.sh --minutes 15
#   WALLET=bc1q... ./scripts/pool-accept-check.sh --pool herominers --diff 32
#   ./scripts/pool-accept-check.sh --pool luckypool
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OFFICIAL="$(cd "$ROOT/../official-miner" && pwd)"
CUDA_HOME="${CUDA_HOME:-$HOME/opt/cuda-13.3}"
BYOS_BUILD="${BYOS_BUILD:-$ROOT/build-byos-cuda}"
MINUTES=10
DIFF=64
POOL_NAME=suprnova
PASS=x

while [[ $# -gt 0 ]]; do
  case "$1" in
    --minutes) MINUTES="$2"; shift 2 ;;
    --diff) DIFF="$2"; shift 2 ;;
    --pool) POOL_NAME="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Stratum is raw TCP — HTTP CONNECT proxies (Cursor sandbox) break it with 403.
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy; do
  unset "$v" || true
done

export PATH="$CUDA_HOME/bin:${HOME}/opt/cuda-env/bin:${PATH:-}"
export LD_LIBRARY_PATH="$BYOS_BUILD:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

WALLET_FILE="$ROOT/.pool-wallet-test.txt"
WALLET="${WALLET:-}"
if [[ -z "$WALLET" && -f "$WALLET_FILE" ]]; then
  WALLET="$(head -1 "$WALLET_FILE")"
fi
if [[ -z "$WALLET" ]]; then
  echo "Set WALLET=bc1q... or create $WALLET_FILE" >&2
  exit 1
fi

case "$POOL_NAME" in
  suprnova)
    # Default pool. Diff often ~0.5 → share TTF ~days @ ~3 kh/s.
    POOL='stratum+tcp://qtc.suprnova.cc:5555'
    USER="${WALLET}.qhashbyos"
    PASS=x
    ;;
  herominers)
    # Preferred when up: fixed DIFF (wallet=DIFF) for faster share samples.
    POOL='stratum+tcp://qubitcoin.herominers.com:1195'
    USER="${WALLET}=${DIFF}.qhashbyos"
    PASS=qhashbyos
    ;;
  luckypool)
    # High min-diff — expect few/no shares at ~3 kh/s; ignores static =DIFF.
    POOL='stratum+tcp://qubitcoin.luckypool.io:8610'
    USER="${WALLET}.qhashbyos"
    PASS=x
    ;;
  *)
    echo "unknown pool: $POOL_NAME (herominers|suprnova|luckypool)" >&2
    exit 1
    ;;
esac

BIN="$OFFICIAL/qubitcoin-miner"
if [[ ! -x "$BIN" ]]; then
  echo "missing $BIN — run ./scripts/build-official-byos.sh --cuda" >&2
  exit 1
fi

echo "== GPU =="
nvidia-smi -L
echo "== self-test =="
"$ROOT/build/qhash-miner" --self-test | tee /tmp/qhash-pool-selftest.txt
if ! grep -q 'CUDA available: yes' /tmp/qhash-pool-selftest.txt; then
  echo "CUDA not available — aborting (would use slow CPU fallback)" >&2
  exit 1
fi

LOG="$ROOT/pool-accept-${POOL_NAME}.log"
SECS=$((MINUTES * 60))
echo "== mining ${MINUTES}m @ $POOL =="
echo "user=$USER  log=$LOG"
timeout "$SECS" "$BIN" -a qhash -o "$POOL" -u "$USER" -p "$PASS" -t 1 2>&1 | tee "$LOG" || true

echo
echo "== accept-rate summary =="
ACCEPT=$(grep -cE 'Accepted|accepted' "$LOG" || true)
REJECT=$(grep -cE 'Rejected|rejected' "$LOG" || true)
STALE=$(grep -cE '[Ss]tale' "$LOG" || true)
echo "accepted_lines=$ACCEPT rejected_lines=$REJECT stale_lines=$STALE"
rg -n 'Accepted|Rejected|Stale|Total:|diff' "$LOG" | tail -40 || true
echo "log: $LOG"
