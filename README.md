# vgpu for PowerVR

Unified Vulkan GPU inference package combining vgpu Python CLI with a
PowerVR-optimized llama.cpp backend.

## Install

```bash
curl -sSL https://raw.githubusercontent.com/voznyiarsen/powervr-package/main/scripts/install.sh | bash
```

Downloads the latest pre-built release and installs it to `~/.local/`.
Add `~/.local/bin` to your `PATH` if not already there.

Requires: Python 3.7+, Vulkan loader, and the Vulkan ICD for your GPU.

## Build from Source

```bash
# Dependencies: cmake, python3, gcc/clang, libvulkan-dev, glslc
bash scripts/setup.sh
bash scripts/build.sh all
bash scripts/test.sh
bash scripts/package.sh
```

## Output

After `build.sh all`, the `dist/` directory contains:

- `dist/bin/llama-server` — LLM inference backend
- `dist/bin/vgpu` — Python CLI entry point
- `dist/vgpu/` — Python modules
