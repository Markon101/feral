#!/data/data/com.termux/files/usr/bin/sh
# Process-local Android runtime setup; does not replace drivers or global ICD files.
set -eu
FERAL_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ -f /vendor/lib64/libOpenCL.so ] && [ -f /system/lib64/libc++.so ]; then
  export FERAL_DECODER_LD_LIBRARY_PATH="${FERAL_DECODER_LD_LIBRARY_PATH-${LD_LIBRARY_PATH-}}"
  export FERAL_DECODER_LD_PRELOAD="${FERAL_DECODER_LD_PRELOAD-${LD_PRELOAD-}}"
  export LD_LIBRARY_PATH="/vendor/lib64:/vendor/lib64/egl:/system/lib64:/data/data/com.termux/files/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export LD_PRELOAD="/system/lib64/libc++.so${LD_PRELOAD:+:$LD_PRELOAD}"
  export OPENCL_DYLIB_PATH="${OPENCL_DYLIB_PATH:-/vendor/lib64/libOpenCL.so}"
fi
if [ "${1:-}" = "--exec" ]; then
  shift
  exec "$@"
fi
exec "$FERAL_ROOT/target/release/feral" "$@"
