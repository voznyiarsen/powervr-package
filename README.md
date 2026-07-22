# vgpu for PowerVR

Unified Vulkan GPU inference package combining vgpu Python CLI with a
PowerVR-optimized llama.cpp backend.

## Quick Start

```bash
# Ensure dependencies are met
bash scripts/setup.sh

# Build everything (llama-server + Python CLI)
bash scripts/build.sh all

# Run tests
bash scripts/test.sh

# Package for distribution
bash scripts/package.sh
```

## Output

After `build.sh all`, the `dist/` directory contains:

- `dist/bin/llama-server` — LLM inference backend
- `dist/bin/vgpu` — Python CLI entry point
- `dist/vgpu/` — Python modules
