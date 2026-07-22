#!/bin/bash
# build.sh - Master build script for vgpu package
#
# Wires together:
#   1. vulkan-wirings (Python CLI + gateway for Vulkan GPU inference)
#   2. llama.cpp-ggml-org (CMake-based LLM inference backend)
#
# Usage: bash scripts/build.sh [target]
#   where target is one of: all, llama, vgpu, clean
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LLAMA_SRC="${PROJECT_ROOT}/../llama.cpp-ggml-org"
VGPU_SRC="${PROJECT_ROOT}/vulkan-wirings"
BUILD_DIR="${PROJECT_ROOT}/build"
DIST_DIR="${PROJECT_ROOT}/dist"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
  echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
  echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $*" >&2
}

# Verify dependencies exist
check_sources() {
  if [ ! -d "$LLAMA_SRC" ]; then
    log_error "llama.cpp source not found at: $LLAMA_SRC"
    exit 1
  fi
  if [ ! -d "$VGPU_SRC" ]; then
    log_error "vulkan-wirings source not found at: $VGPU_SRC"
    exit 1
  fi
  log_info "Sources verified: llama.cpp and vulkan-wirings found"
}

# Build llama.cpp backend
build_llama() {
  log_info "Building llama.cpp backend..."
  
  # Create build directory
  mkdir -p "${BUILD_DIR}/llama"
  cd "${BUILD_DIR}/llama"
  
  # Configure CMake for Vulkan support
  local CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"
  
  # Enable Vulkan backend if available
  if command -v vulkaninfo &>/dev/null || [ -n "${VULKAN_SDK:-}" ]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DGGML_VULKAN=ON"
    log_info "Vulkan support: ENABLED"
  else
    log_warn "Vulkan not detected; building CPU-only backend"
  fi
  
  # Enable Android/Termux detection
  if grep -q "com.termux" /etc/hostname 2>/dev/null || [ -n "${TERMUX_VERSION:-}" ]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DANDROID=1"
    log_info "Android/Termux environment detected"
  fi
  
  # Run CMake
  cmake "${CMAKE_ARGS}" "$LLAMA_SRC"
  
  # Build with available parallelism
  local JOBS=$(nproc 2>/dev/null || echo 1)
  make -j"${JOBS}"
  
  log_info "llama.cpp build complete"
}

# Build vgpu Python package
build_vgpu() {
  log_info "Building vgpu Python package..."
  
  mkdir -p "${DIST_DIR}"
  
  # Copy Python modules
  mkdir -p "${DIST_DIR}/vgpu"
  cp "${VGPU_SRC}/vgpu.py" "${DIST_DIR}/vgpu/cli.py"
  cp "${VGPU_SRC}/vgpu_core.py" "${DIST_DIR}/vgpu/core.py"
  cp "${VGPU_SRC}/vgpu_gateway.py" "${DIST_DIR}/vgpu/gateway.py"
  cp "${VGPU_SRC}/vgpu_chat.py" "${DIST_DIR}/vgpu/chat.py"
  
  # Copy C source and binary
  cp "${VGPU_SRC}/gpu_probe.c" "${DIST_DIR}/vgpu/"
  if [ -f "${VGPU_SRC}/gpu_probe" ]; then
    cp "${VGPU_SRC}/gpu_probe" "${DIST_DIR}/vgpu/"
  fi
  
  # Create __init__.py
  cat > "${DIST_DIR}/vgpu/__init__.py" << 'EOF'
"""vgpu - Vulkan GPU inference CLI with llama.cpp backend."""

__version__ = "1.0.0"
__all__ = ["cli", "core", "gateway", "chat"]
EOF

  # Create entry point
  mkdir -p "${DIST_DIR}/bin"
  cat > "${DIST_DIR}/bin/vgpu" << 'EOF'
#!/usr/bin/env python3
import sys
from vgpu.cli import main

if __name__ == "__main__":
    sys.exit(main())
EOF
  chmod +x "${DIST_DIR}/bin/vgpu"
  
  log_info "vgpu Python package built"
}

# Copy llama-server binary to distribution
wire_llama_server() {
  log_info "Wiring llama-server to distribution..."
  
  local LLAMA_SERVER="${BUILD_DIR}/llama/bin/llama-server"
  
  if [ -f "$LLAMA_SERVER" ]; then
    cp "$LLAMA_SERVER" "${DIST_DIR}/bin/"
    chmod +x "${DIST_DIR}/bin/llama-server"
    log_info "llama-server wired to dist/bin/"
  else
    log_warn "llama-server not found at $LLAMA_SERVER; it may not have been built"
  fi
}

# Create wrapper scripts
create_wrappers() {
  log_info "Creating wrapper scripts..."
  
  mkdir -p "${DIST_DIR}/bin"
  
  # Main vgpu entry point (bash wrapper for compatibility)
  cat > "${DIST_DIR}/bin/vgpu.sh" << 'EOF'
#!/bin/bash
# vgpu - Unified launcher for Vulkan GPU inference
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Prefer Python 3
if command -v python3 >/dev/null 2>&1; then
  exec python3 "$PROJECT_ROOT/bin/vgpu" "$@"
elif command -v python >/dev/null 2>&1; then
  exec python "$PROJECT_ROOT/bin/vgpu" "$@"
else
  echo "error: python3 is required" >&2
  exit 1
fi
EOF
  chmod +x "${DIST_DIR}/bin/vgpu.sh"
  
  log_info "Wrapper scripts created"
}

# Generate build metadata
generate_metadata() {
  log_info "Generating build metadata..."
  
  cat > "${DIST_DIR}/BUILD_INFO.txt" << EOF
vgpu Package - Build Information
=================================

Build Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')
Build Host: $(hostname 2>/dev/null || echo 'unknown')
Uname: $(uname -a)

Components:
  - llama.cpp: $LLAMA_SRC
  - vulkan-wirings: $VGPU_SRC

Build Configuration:
  Python: $(python3 --version 2>&1 || echo 'not found')
  CMake: $(cmake --version 2>&1 | head -1 || echo 'not found')
  
Backend:
  llama-server: $([ -f "${DIST_DIR}/bin/llama-server" ] && file "${DIST_DIR}/bin/llama-server" || echo 'not built')
  vgpu: $([ -f "${DIST_DIR}/bin/vgpu" ] && echo 'python module' || echo 'not found')
EOF
  
  log_info "Metadata written to dist/BUILD_INFO.txt"
}

# Main entry point
main() {
  local TARGET="${1:-all}"
  
  case "$TARGET" in
    all)
      check_sources
      build_llama
      build_vgpu
      wire_llama_server
      create_wrappers
      generate_metadata
      log_info "✓ Full build complete. Distribution in: $DIST_DIR"
      ;;
    llama)
      check_sources
      build_llama
      wire_llama_server
      log_info "✓ llama.cpp build complete"
      ;;
    vgpu)
      check_sources
      build_vgpu
      create_wrappers
      log_info "✓ vgpu build complete"
      ;;
    clean)
      log_info "Cleaning build artifacts..."
      rm -rf "$BUILD_DIR" "$DIST_DIR"
      log_info "✓ Clean complete"
      ;;
    *)
      echo "Usage: $0 {all|llama|vgpu|clean}"
      exit 1
      ;;
  esac
}

main "$@"
