# Exposing the GPU (OpenCL + Vulkan) on Android / Termux

> Device used for verification: Android 17 (aarch64), Termux, Python 3.14.
> GPU: **PowerVR D-Series DXT-48-1536** (Imagination), `ro.hardware.vulkan=powervr`.

---

## 0. Mental model — why this isn't just "install a driver"

On desktop Linux the GPU is reached through standard library paths and an ICD
manifest, and any process can `dlopen` them. **Android is different:**

- The GPU is reached through **vendor HAL libraries** under `/vendor/lib64/...`
  (and `/system/lib64/...`).
- Since Android 7, the dynamic linker enforces **linker namespaces**. A normal
  app/Termux process may only `dlopen` libraries that are either:
  - **(a)** listed in `/system/etc/public.libraries.txt` (the NDK public set), or
  - **(b)** reachable via `LD_LIBRARY_PATH` + present on disk.
- The Vulkan driver is **not** discovered by a `.json` manifest like on desktop;
  Android's `libvulkan.so` **auto-discovers the driver via the HAL mechanism**
  using the `ro.hardware.vulkan` system property.
- `SELinux` can still block a non-system app from loading vendor HALs — that
  surfaces as `avc: denied` in `logcat`/`dmesg`.

So "exposing the GPU" = **(1)** making the loader *find* the vendor library,
**(2)** satisfying its *dependencies*, and **(3)** ensuring the *driver discovery*
path reaches the hardware.

### What we confirmed on *this* device

| Item | Value | Public NDK lib? |
|---|---|---|
| SoC codename | `blazer` (`ro.hardware`) | — |
| GPU | **PowerVR D-Series DXT-48-1536** (Imagination) | — |
| `ro.hardware.vulkan` | `powervr` | — |
| OpenCL loader | `/vendor/lib64/egl/libOpenCL.so` (+ `libOpenCL-pixel.so`) | **No** |
| Vulkan loader | `/system/lib64/libvulkan.so` | **Yes** |
| Vulkan driver | `/vendor/lib64/hw/vulkan.powervr.so` | — (HAL) |

Because `libvulkan.so` is public, **Vulkan needs almost nothing**. Because
`libOpenCL.so` is *not* public, **OpenCL needs `LD_LIBRARY_PATH`**.

---

## PART A — Expose OpenCL

### Step A1. Confirm the OpenCL library exists
```bash
ls -l /vendor/lib64/egl/libOpenCL.so /vendor/lib64/egl/libOpenCL-pixel.so
getprop ro.hardware.vulkan      # -> powervr  (tells you the vendor)
```
If `libOpenCL.so` is missing entirely, **OpenCL is not shipped on this device** —
skip to Vulkan (most phones omit OpenCL). Pixel/Imagination here does ship it.

### Step A2. Understand why a naive `dlopen` fails
```bash
python3 -c "import ctypes; ctypes.CDLL('/vendor/lib64/egl/libOpenCL.so')"
# OSError: dlopen failed: library ".../libOpenCL.so" not found
```
The absolute path "not found" actually means a **dependency** of `libOpenCL.so`
wasn't on the search path. Inspect its `NEEDED` entries:
```bash
llvm-readelf -d /vendor/lib64/egl/libOpenCL.so | grep -i needed
```
You'll typically see it needs other vendor/system libs (e.g. `libcutils`,
`liblog`, `libhardware`, `libz`, EGL/UI libs). Those live in `/vendor/lib64`,
`/system/lib64`, etc.

### Step A3. Add the directories to `LD_LIBRARY_PATH`
The fix is to put the driver dir **and its dependency dirs** on the loader
path, then load by **soname** (not absolute path):
```bash
export LD_LIBRARY_PATH=/vendor/lib64/egl:/vendor/lib64:/system/lib64:/system/lib64/hw
python3 -c "import ctypes; ctypes.CDLL('libOpenCL.so'); print('OpenCL OK')"
# -> OpenCL OK
```
> Why soname and not the absolute path? Loading by absolute path still fails
> because the *loader* resolves the library's own `NEEDED` entries using
> `LD_LIBRARY_PATH`; by soname the path is searched too. With `LD_LIBRARY_PATH`
> set, both work — soname is the robust choice.

### Step A4. Query platforms/devices (verify the GPU is reachable)
```python
import ctypes
cl = ctypes.CDLL("libOpenCL.so")
n = ctypes.c_uint32(0)
cl.clGetPlatformIDs(0, None, ctypes.byref(n))
print("platforms:", n.value)   # -> 1  (OpenCL 3.0 / PowerVR / Imagination)
```
The full, runnable benchmark is at `gpu_opencl_test.py` (runs the kernel,
reports ~183 GFLOP/s, verifies vs CPU).

### Step A5. Make it persistent (optional)
Add to `~/.bashrc` / `~/.profile`, or use the wrapper (see Part C):
```bash
echo 'export LD_LIBRARY_PATH=/vendor/lib64/egl:/vendor/lib64:/system/lib64' >> ~/.bashrc
```

---

## PART B — Expose Vulkan

### Step B1. Locate loader + driver
```bash
ls -l /system/lib64/libvulkan.so            # the loader (public NDK lib)
ls -l /vendor/lib64/hw/vulkan.*.so          # the vendor driver (HAL module)
getprop ro.hardware.vulkan                  # -> powervr  (HAL module id)
```
Confirm `libvulkan.so` is allowed for apps:
```bash
grep -i vulkan /system/etc/public.libraries.txt
# -> libvulkan.so   (means apps can load it directly)
```

### Step B2. (Usually) just load it — no `LD_LIBRARY_PATH` required
```bash
python3 -c "import ctypes; ctypes.CDLL('libvulkan.so'); print('Vulkan OK')"
# -> Vulkan OK
```
Because it's a public lib *and* the driver is found by the HAL mechanism,
**no environment variable is needed** on Android. (Setting `LD_LIBRARY_PATH`
anyway is harmless and helps if your app also links other vendor libs.)

### Step B3. The driver-discovery path (critical concept)
On **desktop** Vulkan you must point the loader at a manifest:
```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/foo.json   # desktop only
```
On **Android**, do **NOT** set that. The system `libvulkan.so` calls
`hw_get_module("vulkan", ...)` and picks `vulkan.<ro.hardware.vulkan>.so`
(`vulkan.powervr.so`) from `/vendor/lib64/hw/`. No JSON exists on this device
(we checked — `find` for `*icd*.json` returned nothing). So leave
`VK_ICD_FILENAMES` empty unless you deliberately install a desktop/mesa loader.

### Step B4. Enumerate the physical GPU (verify)
The runnable proof is `vulkan_enum.py`. It creates a `VkInstance` and calls
`vkEnumeratePhysicalDevices`:
```bash
python3 vulkan_enum.py
# Vulkan instance created (VkResult 0)
# Physical devices reported: 1
#   GPU[0]: PowerVR D-Series DXT-48-1536 MC1
# RESULT: Vulkan exposed the PowerVR GPU to Termux.
```
It worked **with and without** `LD_LIBRARY_PATH`, confirming Vulkan needs no
special exposure on this device.

---

## PART C — One-command reusable setup (`gpu_env.sh`)

`gpu_env.sh` makes exposure reproducible:
```bash
# Export into the current shell:
. ./gpu_env.sh

# OR run a command with the env set (preferred for scripts):
./gpu_env.sh python3 gpu_opencl_test.py
./gpu_env.sh python3 vulkan_enum.py
```
What it does:
- Sets `LD_LIBRARY_PATH` to the union of OpenCL/Vulkan loader + dependency dirs.
- Leaves `VK_ICD_FILENAMES` empty (correct for Android's HAL discovery).
- Exports `ANDROID_GPU_VENDOR` / `ANDROID_GPU_HW` for your scripts.

Verified output:
```
GPU env set: vendor=powervr hw=blazer
OpenCL : libOpenCL.so loaded OK
Vulkan : libvulkan.so loaded OK
```

`gpu_env.sh` contents:
```sh
#!/bin/sh
# gpu_env.sh — expose the PowerVR GPU (OpenCL + Vulkan) to Termux processes.
# Usage:
#   . ./gpu_env.sh                           # export into current shell
#   ./gpu_env.sh python3 vulkan_enum.py      # run a command with env set

GPU_LIBDIRS="/vendor/lib64/egl:/vendor/lib64/hw:/vendor/lib64:/system/lib64/hw:/system/lib64"
export LD_LIBRARY_PATH="${GPU_LIBDIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Android system loader auto-discovers the driver via ro.hardware.vulkan;
# no VK_ICD_FILENAMES needed. Only set it for a desktop/mesa loader + .json.
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-}"

export ANDROID_GPU_VENDOR="$(getprop ro.hardware.vulkan 2>/dev/null)"
export ANDROID_GPU_HW="$(getprop ro.hardware 2>/dev/null)"

echo "GPU env set: vendor=${ANDROID_GPU_VENDOR:-?} hw=${ANDROID_GPU_HW:-?}"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

if [ $# -gt 0 ]; then
  exec "$@"
fi
```

---

## PART D — Troubleshooting matrix

| Symptom | Cause | Fix |
|---|---|---|
| `dlopen failed: ... not found` (absolute path) | Dependency not on search path | Set `LD_LIBRARY_PATH` to driver **+ dependency** dirs; load by **soname** |
| `dlopen failed` even with path | Library not in public namespace *and* not reachable | Add its dir to `LD_LIBRARY_PATH`; if still blocked, the device hard-restricts vendor libs |
| `libOpenCL.so` doesn't exist | OEM didn't ship OpenCL | Use **Vulkan** instead (or install software OpenCL like PoCL — usually not feasible on Android) |
| Vulkan instance OK but **0 physical devices** | Driver HAL not reachable / `ro.hardware.vulkan` mismatch | Check `ls /vendor/lib64/hw/vulkan.*.so`; ensure loader variant matches `getprop ro.hardware.vulkan` |
| `avc: denied` in `logcat`/`dmesg` | SELinux blocks loading vendor HAL from app domain | Requires root or a vendor-allowed context; cannot fix from untrusted app alone |
| 32-bit vs 64-bit `ELF class` mismatch | Script/process is 32-bit but libs are `lib64` | Ensure aarch64 Python (`python3 --version` + `uname -m` = aarch64); use `lib64` paths |
| Works in shell but not in a GUI/app | `LD_LIBRARY_PATH` not inherited by the launcher | Wrap the launch: `env LD_LIBRARY_PATH=... am start ...` or set in the app's wrapper |

### Vendor / path cheat-sheet (for other devices)
- **Imagination / Pixel (this device):** `/vendor/lib64/egl/libOpenCL.so`,
  `/vendor/lib64/hw/vulkan.powervr.so`, `ro.hardware.vulkan=powervr`
- **Arm Mali:** `/vendor/lib64/egl/libOpenCL.so` (ships as "libOpenCL" via
  `ro.hardware=vulkan.mali`), Vulkan driver `vulkan.mali.so`
- **Qualcomm Adreno:** `/vendor/lib64/libOpenCL.so`, `ro.hardware.vulkan=adreno`,
  driver `vulkan.adreno.so`
- **NVIDIA Tegra / others:** check `getprop ro.hardware.vulkan` then locate
  `vulkan.<that>.so` under `/vendor/lib64/hw`.

---

## PART E — From "exposed" to "actually using it"

You now have working **enumerations** for both APIs. Natural next steps:

- **OpenCL compute:** already implemented in `gpu_opencl_test.py`
  (context → program → kernel → NDRange → readback → verify). Swap the kernel
  string for your own workload.
- **Vulkan compute:** build a `VkDevice` + `VkShaderModule` (SPIR-V compute
  shader) + `vkQueueSubmit`. The `vulkan` PyPI package won't install here (no
  wheel for py3.14/aarch64), so either keep using `ctypes` wrappers or compile
  a small C program against the NDK. For **graphics**, create a surface via
  `VkAndroidSurfaceCreateInfoKHR` (requires an Android `ANativeWindow` — only
  from a real app, not pure Termux/Python).
- **Cross-check:** run both `gpu_opencl_test.py` and `vulkan_enum.py` through
  `./gpu_env.sh` to confirm both paths hit the same physical GPU.

**Bottom line:** Vulkan is exposed for free (public loader + HAL auto-discovery).
OpenCL needs `LD_LIBRARY_PATH` pointed at `/vendor/lib64/egl` (+ dependency
dirs) because it isn't a public NDK library. Both are verified working against
the PowerVR DXT-48-1536 on this device.
