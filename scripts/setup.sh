#!/bin/bash
# setup.sh - Environment setup and dependency checking for vgpu package
#
# Checks that all required tools and libraries are available for building
# and running the vgpu + llama.cpp unified package.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[✓]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[⚠]${NC} $*"; }
log_error() { echo -e "${RED}[✗]${NC} $*" >&2; }
log_section() { echo -e "\n${CYAN}== $* ==${NC}"; }

# Check for required commands
check_command() {
  local cmd="$1"
  local install_hint="${2:-'install from your package manager'}"
  
  if command -v "$cmd" &>/dev/null; then
    log_info "Found: $cmd ($(command -v "$cmd"))"
    return 0
  else
    log_error "Missing: $cmd ($install_hint)"
    return 1
  fi
}

# Check for optional commands
check_optional() {
  local cmd="$1"
  local reason="${2:-''}"
  
  if command -v "$cmd" &>/dev/null; then
    log_info "Found (optional): $cmd"
    return 0
  else
    if [ -n "$reason" ]; then
      log_warn "Optional: $cmd not found ($reason)"
    else
      log_warn "Optional: $cmd not found"
    fi
    return 1
  fi
}

# Check Python environment
check_python() {
  log_section "Python Environment"
  
  if ! check_command "python3" "install python3"; then
    log_error "python3 is required for vgpu"
    return 1
  fi
  
  local PY_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
  log_info "Python version: $PY_VERSION"
  
  # Check for venv (needed for isolation)
  check_optional "venv" "for virtual environments (install python3-venv)"
  
  return 0
}

# Check C/C++ toolchain
check_toolchain() {
  log_section "C/C++ Build Toolchain"
  
  local required_missing=0
  
  if ! check_command "cmake" "install cmake (v3.14+)"; then
    required_missing=$((required_missing + 1))
  else
    local CMAKE_VERSION=$(cmake --version | head -1)
    log_info "$CMAKE_VERSION"
  fi
  
  if ! check_command "gcc" "install gcc"; then
    required_missing=$((required_missing + 1))
  fi
  
  check_optional "g++" "needed for C++ parts of llama.cpp"
  check_optional "clang" "alternative C compiler"
  
  return $required_missing
}

# Check for GPU/Vulkan support
check_vulkan() {
  log_section "Vulkan Support (Optional but Recommended)"
  
  check_optional "vulkaninfo" "run 'vgpu gpu' to probe devices at runtime"
  
  if [ -d "/vendor/lib64/hw" ]; then
    log_info "Android hardware abstraction layer detected"
  fi
  
  if [ -n "${VULKAN_SDK:-}" ]; then
    log_info "VULKAN_SDK set: $VULKAN_SDK"
  fi
  
  if [ -d "/usr/include/vulkan" ] || [ -d "/usr/local/include/vulkan" ]; then
    log_info "Vulkan headers found"
  else
    log_warn "Vulkan headers not found (Vulkan backend may not build)"
  fi
}

# Check for optional build tools
check_optional_tools() {
  log_section "Optional Build Tools"
  
  check_optional "make" "for parallel builds (install make)"
  check_optional "ninja" "faster build backend (install ninja-build)"
  check_optional "ccache" "for faster rebuilds (install ccache)"
  check_optional "git" "for version info in builds"
}

# Check system libraries
check_system_libs() {
  log_section "System Libraries"
  
  # Check for common development headers
  if [ -f "/usr/include/stdio.h" ]; then
    log_info "Standard C headers found"
  else
    log_warn "Standard C headers not found; your system may not have build essentials"
  fi
  
  # Termux/Android checks
  if [ -n "${TERMUX_VERSION:-}" ]; then
    log_info "Running in Termux (Android)"
    
    check_optional "pkg" "Termux package manager"
    
    if grep -q "com.termux" /etc/hostname 2>/dev/null || [ -f "$PREFIX/etc/os-release" ] 2>/dev/null; then
      log_info "Termux environment variables detected"
    fi
  fi
}

# Check source directories
check_sources() {
  log_section "Source Directories"
  
  local LLAMA_SRC="${PROJECT_ROOT}/../llama.cpp-ggml-org"
  local VGPU_SRC="${PROJECT_ROOT}/vulkan-wirings"
  
  if [ -d "$LLAMA_SRC" ]; then
    log_info "llama.cpp found: $LLAMA_SRC"
  else
    log_error "llama.cpp not found: $LLAMA_SRC"
    return 1
  fi
  
  if [ -d "$VGPU_SRC" ]; then
    log_info "vulkan-wirings found: $VGPU_SRC"
  else
    log_error "vulkan-wirings not found: $VGPU_SRC"
    return 1
  fi
  
  return 0
}

# Show build commands
show_next_steps() {
  log_section "Next Steps"
  
  echo "To build the vgpu package:"
  echo ""
  echo "  cd $PROJECT_ROOT"
  echo "  bash scripts/build.sh all          # Build everything"
  echo "  bash scripts/build.sh llama        # Build just llama.cpp"
  echo "  bash scripts/build.sh vgpu         # Build just vgpu"
  echo "  bash scripts/build.sh clean        # Clean build artifacts"
  echo ""
  echo "Or use npm/yarn:"
  echo "  npm run build"
  echo "  npm run setup"
  echo ""
}

# Main
main() {
  local failures=0
  
  check_python || failures=$((failures + 1))
  check_toolchain || failures=$((failures + 1))
  check_vulkan
  check_optional_tools
  check_system_libs
  check_sources || failures=$((failures + 1))
  
  log_section "Summary"
  
  if [ $failures -eq 0 ]; then
    log_info "All required dependencies found! Ready to build."
    show_next_steps
    return 0
  else
    log_error "$failures required component(s) missing"
    show_next_steps
    return 1
  fi
}

main "$@"
