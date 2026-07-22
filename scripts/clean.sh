#!/bin/bash
# clean.sh - Clean build artifacts
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Cleaning vgpu build artifacts..."

rm -rf "$PROJECT_ROOT/build"
rm -rf "$PROJECT_ROOT/dist"
rm -rf "$PROJECT_ROOT/release"
rm -rf "$PROJECT_ROOT/__pycache__"
find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$PROJECT_ROOT" -type f -name "*.pyc" -delete 2>/dev/null || true
find "$PROJECT_ROOT" -type f -name "*.egg-info" -delete 2>/dev/null || true

echo "✓ Clean complete"
