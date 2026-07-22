#!/bin/bash
# install.sh - Download and install the latest vgpu release
# Usage: curl -sSL https://raw.githubusercontent.com/voznyiarsen/powervr-package/main/scripts/install.sh | bash
set -euo pipefail

REPO="voznyiarsen/powervr-package"
API="https://api.github.com/repos/$REPO"

detect_platform() {
  if [ -n "${TERMUX_VERSION:-}" ] || [ -d "/data/data/com.termux" ]; then
    echo "android-termux"
  else
    echo "linux"
  fi
}

detect_arch() {
  local arch
  arch=$(uname -m)
  case "$arch" in
    x86_64|amd64) echo "x86_64" ;;
    aarch64|arm64) echo "aarch64" ;;
    armv7l|armv8l) echo "armv7l" ;;
    *) echo "$arch" ;;
  esac
}

detect_tag() {
  if [ -n "${VGPU_TAG:-}" ]; then
    echo "$VGPU_TAG"
    return
  fi
  curl -sL "$API/releases/tags/continuous" | grep -o '"tag_name": *"[^"]*"' | head -1 | cut -d'"' -f4
}

PLATFORM=$(detect_platform)
ARCH=$(detect_arch)
TAG=$(detect_tag)

if [ -z "$TAG" ]; then
  echo "error: could not determine latest release tag" >&2
  exit 1
fi

PREFIX="${VGPU_PREFIX:-$HOME/.local}"
BINDIR="$PREFIX/bin"
LIBDIR="$PREFIX/share/powervr"

case "$ARCH" in
  x86_64)  DL_ARCH="x86_64" ;;
  aarch64) DL_ARCH="aarch64" ;;
  armv7l)  DL_ARCH="armv7l" ;;
  *)       DL_ARCH="$ARCH" ;;
esac

case "$PLATFORM" in
  android-termux)  DL_PLATFORM="linux" ;;
  linux)           DL_PLATFORM="linux" ;;
  darwin)          DL_PLATFORM="macos" ;;
  *)               DL_PLATFORM="$PLATFORM" ;;
esac

FILENAME="vgpu-1.0.0-${DL_PLATFORM}-${DL_ARCH}.tar.gz"
DOWNLOAD_URL="https://github.com/$REPO/releases/download/$TAG/$FILENAME"

echo "Platform:   $PLATFORM ($DL_PLATFORM)"
echo "Arch:       $ARCH ($DL_ARCH)"
echo "Tag:        $TAG"
echo "Download:   $DOWNLOAD_URL"
echo "Install to: $LIBDIR"
echo ""

TMPDIR=$(mktemp -d)
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

echo "Downloading..."
curl -sL "$DOWNLOAD_URL" -o "$TMPDIR/vgpu.tar.gz"

echo "Extracting..."
mkdir -p "$TMPDIR/dist"
tar -xzf "$TMPDIR/vgpu.tar.gz" -C "$TMPDIR/dist"

rm -rf "$LIBDIR"
mkdir -p "$LIBDIR" "$BINDIR"

cp -r "$TMPDIR/dist/dist"/* "$LIBDIR/"

for entry in "$LIBDIR"/bin/*; do
  if [ -f "$entry" ]; then
    name=$(basename "$entry")
    ln -sf "$entry" "$BINDIR/$name"
    chmod +x "$entry"
  fi
done

echo ""
echo " Installed to $LIBDIR"
echo " Symlinks in $BINDIR:"
ls -1 "$LIBDIR/bin/" 2>/dev/null | sed 's/^/   /'

if ! echo "$PATH" | tr ':' '\n' | grep -qxF "$BINDIR" 2>/dev/null; then
  echo ""
  echo " Add to your shell:"
  echo "   export PATH=\"\$PATH:$BINDIR\""
fi

echo ""
echo " Quick test: vgpu --help"
