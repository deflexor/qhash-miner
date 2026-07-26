#!/usr/bin/env bash
# Build official-miner with qhash-miner BYOS (CPU or CUDA).
# Usage:
#   ./scripts/build-official-byos.sh           # CPU BYOS (stock FP32)
#   ./scripts/build-official-byos.sh --cuda    # CUDA scanhash + FP64
#   ./scripts/build-official-byos.sh --fp64    # CPU BYOS FP64 consensus
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OFFICIAL="$(cd "$ROOT/../official-miner" && pwd)"
DEPS="${MINER_DEPS:-$HOME/opt/miner-deps}"
CUDA_HOME="${CUDA_HOME:-$HOME/opt/cuda-13.3}"
DEPS_BIN="$ROOT/.deps-bin"

MODE=cpu
FP64=0
for arg in "$@"; do
  case "$arg" in
    --cuda) MODE=cuda; FP64=1 ;;
    --fp64) FP64=1 ;;
    --cpu)  MODE=cpu ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

# Unversioned wrappers (extracted Ubuntu packages use -1.18 suffix).
mkdir -p "$DEPS_BIN"
ln -sfn "${DEPS}/usr/bin/aclocal-1.18" "$DEPS_BIN/aclocal"
cat > "$DEPS_BIN/automake" <<EOF
#!/bin/sh
exec "${DEPS}/usr/bin/automake-1.18" --libdir="${DEPS}/usr/share/automake-1.18" "\$@"
EOF
chmod +x "$DEPS_BIN/automake"
ln -sfn "${DEPS}/usr/bin/autoconf" "$DEPS_BIN/autoconf"
ln -sfn "${DEPS}/usr/bin/autoheader" "$DEPS_BIN/autoheader"
ln -sfn "${DEPS}/usr/bin/autom4te" "$DEPS_BIN/autom4te"
ln -sfn "${DEPS}/usr/bin/libtoolize" "$DEPS_BIN/libtoolize"
ln -sfn "${DEPS}/usr/bin/m4" "$DEPS_BIN/m4"

export PATH="$DEPS_BIN:$CUDA_HOME/bin:${DEPS}/usr/bin:${PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${DEPS}/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="${DEPS}/usr/lib/x86_64-linux-gnu/pkgconfig:${DEPS}/usr/share/pkgconfig:${PKG_CONFIG_PATH:-}"
export CPATH="${DEPS}/usr/include/x86_64-linux-gnu:${DEPS}/usr/include:${CPATH:-}"
export LIBRARY_PATH="${DEPS}/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export CUDA_PATH="$CUDA_HOME"
export M4="${DEPS}/usr/bin/m4"

# Workspace-local autoconf datadir (autom4te.cfg hardcodes /usr/share/autoconf).
AC_DATA="$ROOT/.deps-autoconf"
if [[ ! -f "$AC_DATA/autom4te.cfg" ]] || ! grep -q "$AC_DATA" "$AC_DATA/autom4te.cfg" 2>/dev/null; then
  rm -rf "$AC_DATA"
  mkdir -p "$AC_DATA"
  cp -a "${DEPS}/usr/share/autoconf/." "$AC_DATA/"
  sed -i "s|/usr/share/autoconf|$AC_DATA|g" "$AC_DATA/autom4te.cfg"
fi
export autom4te_perllibdir="$AC_DATA"
export AC_MACRODIR="$AC_DATA"
export PERL5LIB="${DEPS}/usr/share/automake-1.18:${AC_DATA}${PERL5LIB:+:$PERL5LIB}"
export ACLOCAL_PATH="${DEPS}/usr/share/aclocal${ACLOCAL_PATH:+:$ACLOCAL_PATH}"
export ACLOCAL_AUTOMAKE_DIR="${DEPS}/usr/share/aclocal-1.18"
export AUTOM4TE="${DEPS_BIN}/autom4te"
export AUTOCONF="${DEPS_BIN}/autoconf"
export AUTOHEADER="${DEPS_BIN}/autoheader"
export AUTOMAKE="${DEPS_BIN}/automake"
export ACLOCAL="${DEPS_BIN}/aclocal"
export LIBTOOLIZE="${DEPS_BIN}/libtoolize"

BYOS_BUILD="$ROOT/build-byos-${MODE}"
CMAKE_ARGS=(
  -S "$ROOT" -B "$BYOS_BUILD"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"
)
if [[ "$MODE" == "cpu" ]]; then
  CMAKE_ARGS+=(-DQHASH_ENABLE_CUDA=OFF -DQHASH_ENABLE_CUSTATEVEC=OFF)
else
  CMAKE_ARGS+=(-DQHASH_ENABLE_CUDA=ON -DQHASH_ENABLE_CUSTATEVEC=OFF)
fi
if [[ "$FP64" == "1" ]]; then
  CMAKE_ARGS+=(-DQHASH_BYOS_FP64=ON)
fi

echo "==> cmake qhash-miner ($MODE FP64=$FP64)"
cmake "${CMAKE_ARGS[@]}"
cmake --build "$BYOS_BUILD" -j"$(nproc)"

CONF_ARGS=(
  --with-curl="$DEPS/usr"
  --enable-qhash-byos
  --with-qhash-miner="$ROOT"
  --with-qhash-miner-build="$BYOS_BUILD"
)
if [[ "$MODE" == "cuda" ]]; then
  CONF_ARGS+=(--enable-qhash-byos-cuda)
fi
if [[ "$FP64" == "1" ]]; then
  CONF_ARGS+=(--enable-qhash-byos-fp64)
fi

echo "==> autotools official-miner"
cd "$OFFICIAL"
# Ensure automake aux files exist (config.guess/sub from autotools-dev).
for f in config.guess config.sub; do
  if [[ ! -f "$f" || -L "$f" ]]; then
    src="${DEPS}/usr/share/misc/$f"
    if [[ -f "$src" ]]; then
      cp -a "$src" "$f"
    fi
  fi
done
if [[ ! -f configure || configure.ac -nt configure || Makefile.am -nt Makefile.in ]]; then
  ./autogen.sh
fi

# Fresh configure when switching BYOS modes
make distclean >/dev/null 2>&1 || true

# Restore aux scripts after distclean (automake --add-missing may have owned them).
for f in config.guess config.sub; do
  if [[ ! -f "$f" || -L "$f" ]]; then
    src="${DEPS}/usr/share/misc/$f"
    if [[ -f "$src" ]]; then
      cp -a "$src" "$f"
    fi
  fi
done

CFLAGS="-O3 -march=native -Wall" \
CXXFLAGS="-O3 -march=native -Wall" \
CPPFLAGS="-I${DEPS}/usr/include/x86_64-linux-gnu -I${DEPS}/usr/include" \
LDFLAGS="-L${DEPS}/usr/lib/x86_64-linux-gnu" \
LIBCURL="-lcurl" \
./configure "${CONF_ARGS[@]}"

make -j"$(nproc)"

BIN="$OFFICIAL/qubitcoin-miner"
echo "==> built $BIN"
echo "Benchmark: LD_LIBRARY_PATH=$BYOS_BUILD:$CUDA_HOME/lib:\$LD_LIBRARY_PATH \\"
echo "  $BIN -a qhash --benchmark -t 1"
