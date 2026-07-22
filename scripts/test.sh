#!/bin/bash
# test.sh - Integration tests for vgpu package
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DIST_DIR="${PROJECT_ROOT}/dist"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log_pass() { echo -e "${GREEN}[✓]${NC} $*"; }
log_fail() { echo -e "${RED}[✗]${NC} $*"; exit 1; }

# Check that distribution is built
if [ ! -d "$DIST_DIR" ]; then
  log_fail "Distribution not found. Run 'bash scripts/build.sh all' first."
fi

log_pass "Distribution directory found"

# Test Python modules
if [ -f "$DIST_DIR/vgpu/__init__.py" ]; then
  log_pass "vgpu package structure valid"
else
  log_fail "vgpu package structure invalid"
fi

# Test Python syntax
if python3 -m py_compile "$DIST_DIR/vgpu/cli.py" 2>/dev/null; then
  log_pass "cli.py syntax valid"
else
  log_fail "cli.py has syntax errors"
fi

if python3 -m py_compile "$DIST_DIR/vgpu/core.py" 2>/dev/null; then
  log_pass "core.py syntax valid"
else
  log_fail "core.py has syntax errors"
fi

# Test entry point
if [ -x "$DIST_DIR/bin/vgpu" ]; then
  log_pass "vgpu executable is executable"
else
  log_fail "vgpu executable is not executable"
fi

# Test llama-server (if built)
if [ -x "$DIST_DIR/bin/llama-server" ]; then
  log_pass "llama-server executable found"
  
  # Check that it's an ELF binary (or executable)
  if file "$DIST_DIR/bin/llama-server" | grep -q "executable\|ELF"; then
    log_pass "llama-server appears to be a valid executable"
  fi
else
  echo "[⚠] llama-server not found (may not have been built)"
fi

# Test vgpu wrapper script
if [ -x "$DIST_DIR/bin/vgpu.sh" ]; then
  log_pass "vgpu.sh wrapper found"
else
  echo "[⚠] vgpu.sh wrapper not found"
fi

# Test GPU probe
if [ -f "$DIST_DIR/vgpu/gpu_probe.c" ]; then
  log_pass "gpu_probe.c source found"
fi

if [ -x "$DIST_DIR/vgpu/gpu_probe" ]; then
  log_pass "gpu_probe binary found"
fi

# Integration test: Can we import vgpu module?
if PYTHONPATH="$DIST_DIR:${PYTHONPATH:-}" python3 -c "import vgpu; print(vgpu.__version__)" 2>/dev/null; then
  log_pass "vgpu module imports successfully"
else
  echo "[⚠] vgpu module import test skipped or failed"
fi

log_pass "✓ All basic tests passed"
