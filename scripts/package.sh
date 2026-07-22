#!/bin/bash
# package.sh - Create distributable packages of vgpu
#
# Generates:
#   - Tarball for distribution
#   - Python wheel for PyPI
#   - Platform-specific binaries
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DIST_DIR="${PROJECT_ROOT}/dist"
RELEASE_DIR="${PROJECT_ROOT}/release"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# Verify dist directory exists
if [ ! -d "$DIST_DIR" ]; then
  log_error "Distribution not built yet. Run 'bash scripts/build.sh all' first."
  exit 1
fi

# Create release directory
mkdir -p "$RELEASE_DIR"
log_info "Packaging to: $RELEASE_DIR"

# Get version
VERSION=$(grep '"version"' "$PROJECT_ROOT/package.json" | head -1 | sed 's/.*"version": "\([^"]*\)".*/\1/')
PLATFORM=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
TIMESTAMP=$(date +%s)

# Create tarball distribution
create_tarball() {
  local TARBALL="$RELEASE_DIR/vgpu-${VERSION}-${PLATFORM}-${ARCH}.tar.gz"
  
  log_info "Creating tarball: $TARBALL"
  
  cd "$PROJECT_ROOT"
  tar czf "$TARBALL" \
    --exclude='.git' \
    --exclude='build/' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    dist/ \
    package.json \
    pyproject.toml \
    README.md \
    LICENSE
  
  log_info "Tarball created: $TARBALL ($(du -h "$TARBALL" | cut -f1))"
}

# Create Python wheel
create_wheel() {
  if ! command -v python3 &>/dev/null; then
    log_error "python3 not found; skipping wheel creation"
    return 1
  fi
  
  if ! python3 -m pip show wheel &>/dev/null; then
    log_info "Installing build tools..."
    python3 -m pip install --quiet wheel setuptools
  fi
  
  log_info "Building Python wheel..."
  
  cd "$PROJECT_ROOT"
  python3 -m pip wheel . --wheel-dir "$RELEASE_DIR" --no-deps 2>&1 | tail -5
  
  log_info "Wheel packages created in: $RELEASE_DIR"
}

# Create platform-specific binary package
create_binary_package() {
  local BIN_PKG="$RELEASE_DIR/vgpu-${VERSION}-${PLATFORM}-${ARCH}-bin.tar.gz"
  
  log_info "Creating binary package: $BIN_PKG"
  
  # Create a minimal binary-only distribution
  mkdir -p "$RELEASE_DIR/vgpu-bin"
  
  if [ -d "$DIST_DIR/bin" ]; then
    cp -r "$DIST_DIR/bin" "$RELEASE_DIR/vgpu-bin/"
  fi
  
  if [ -d "$DIST_DIR/vgpu" ]; then
    cp -r "$DIST_DIR/vgpu" "$RELEASE_DIR/vgpu-bin/"
  fi
  
  # Add runtime instructions
  cat > "$RELEASE_DIR/vgpu-bin/README.txt" << 'EOF'
vgpu - Vulkan GPU Inference CLI
================================

Quick Start:
  ./bin/vgpu.sh --help
  ./bin/vgpu.sh gpu              # Check for available GPU
  ./bin/vgpu.sh pull <model>     # Download a model
  ./bin/vgpu.sh serve <model>    # Start server

Requirements:
  - Python 3.7+
  - Vulkan loader and headers
  - llama.cpp compiled backend

See https://github.com/... for full documentation
EOF
  
  cd "$RELEASE_DIR"
  tar czf "$BIN_PKG" vgpu-bin/
  rm -rf vgpu-bin
  
  log_info "Binary package created: $BIN_PKG ($(du -h "$BIN_PKG" | cut -f1))"
}

# Create checksums
create_checksums() {
  log_info "Creating checksums..."
  
  cd "$RELEASE_DIR"
  
  # SHA256
  sha256sum vgpu-*.tar.gz > SHA256SUMS 2>/dev/null || true
  if [ -f "SHA256SUMS" ] && [ -s "SHA256SUMS" ]; then
    log_info "SHA256 checksums: $(wc -l < SHA256SUMS) files"
  fi
  
  # Create manifest
  cat > MANIFEST.txt << EOF
vgpu Package Release
====================
Version: $VERSION
Build Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')
Platform: $PLATFORM
Architecture: $ARCH
Host: $(hostname 2>/dev/null || echo 'unknown')

Contents:
EOF
  
  ls -lh vgpu-* >> MANIFEST.txt 2>/dev/null || true
}

# Create documentation
create_docs() {
  log_info "Creating documentation..."
  
  cat > "$RELEASE_DIR/BUILD_GUIDE.md" << 'EOF'
# Building vgpu Package

## Prerequisites

- Python 3.7+
- CMake 3.14+
- C/C++ compiler (gcc/clang)
- Vulkan SDK (optional but recommended)

## Quick Build

```bash
cd vgpu-package
bash scripts/setup.sh    # Check dependencies
bash scripts/build.sh    # Build everything
```

## Building on Android/Termux

```bash
pkg install clang cmake python3 vulkan-loader-android aria2
bash scripts/setup.sh
bash scripts/build.sh all
```

## Building Components Separately

```bash
bash scripts/build.sh llama    # Just llama.cpp
bash scripts/build.sh vgpu     # Just Python CLI
bash scripts/build.sh clean    # Clean artifacts
```

## Output

- `dist/bin/vgpu` - Main CLI
- `dist/bin/llama-server` - Inference backend
- `dist/vgpu/` - Python modules

## Distribution

```bash
bash scripts/package.sh  # Create distributable packages
# Output in ./release/
```

## Usage After Building

```bash
export PATH="$PWD/dist/bin:$PATH"
vgpu gpu                 # Check GPU availability
vgpu pull qwen2.5-0.5b-q4_0
vgpu serve qwen2.5-0.5b-q4_0
```
EOF
  
  log_info "Documentation created in: $RELEASE_DIR/BUILD_GUIDE.md"
}

# Main
main() {
  log_info "Packaging vgpu v$VERSION for $PLATFORM/$ARCH"
  
  create_tarball
  create_binary_package
  create_wheel || log_error "Wheel creation failed (non-critical)"
  create_checksums
  create_docs
  
  log_info "✓ Packaging complete. Artifacts in: $RELEASE_DIR"
  log_info "Files:"
  ls -lh "$RELEASE_DIR" | tail -n +2 | awk '{print "  " $9 " (" $5 ")"}'
}

main "$@"
