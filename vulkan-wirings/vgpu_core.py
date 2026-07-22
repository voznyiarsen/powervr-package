#!/usr/bin/env python3
"""vgpu_core - shared logic for the vgpu CLI and gateway.

Centralizes the pieces that were previously duplicated between the bash CLI
(`vgpu`) and the Python gateway (`vgpu_gateway.py`): color output, the model
registry, the GPU reachability probe, server state (pid/port/marker files),
the GGUF memory-footprint math, the VRAM/OOM gates, and the backend launcher
(with CPU auto-fallback).

Everything here targets the standard library only so the whole project keeps
running with just `bash` + `python3` + `curl` + `aria2c`.
"""

import os
import sys
import glob
import struct
import shutil
import subprocess

# ---------------------------------------------------------------------------
# Paths / constants (match the bash CLI defaults)
# ---------------------------------------------------------------------------
LIBDIR = "/data/data/com.termux/files/usr/lib/ollama"
USRLIB = "/data/data/com.termux/files/usr/lib"
HOME = os.environ.get("HOME", os.path.expanduser("~"))
MODELDIR = os.path.join(HOME, "vgpu_models")
REGISTRY = os.path.join(MODELDIR, "registry.txt")
PIDFILE = os.path.join(MODELDIR, ".server.pid")
PORTFILE = os.path.join(MODELDIR, ".server.port")
MODELFILE = os.path.join(MODELDIR, ".server.model")
BACKEND_PIDFILE = os.path.join(MODELDIR, ".gateway.backend.pid")
SERVE_LOG = os.path.join(MODELDIR, "serve.log")
GATEWAY_LOG = os.path.join(MODELDIR, "gateway.log")

DEFAULT_CTX = 4096
DEFAULT_PORT = int(os.environ.get("PORT", "11434"))
GATEWAY_BK_PORT = 11435

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GPU_PROBE_SRC = os.path.join(SCRIPT_DIR, "gpu_probe.c")
GPU_PROBE_BIN = os.path.join(SCRIPT_DIR, "gpu_probe")

DEFAULT_REGISTRY = """\
qwen2.5-0.5b-q4_0 https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf
qwen2.5-1.5b-q8_0 https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf
qwen2.5-3b-q4_0 https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_0.gguf
qwen2.5-3b-q8_0 https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q8_0.gguf
llama3.2-3b-q8_0 https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf
"""


# ---------------------------------------------------------------------------
# Color (uv-style palette). Resolution precedence (mirrors the bash CLI):
#   explicit VG_COLOR / --no-color > NO_COLOR > FORCE_COLOR/CLICOLOR_FORCE > TTY
# ---------------------------------------------------------------------------
def resolve_color(vg_color="auto"):
    """Return (use_color: bool, set_env: bool)."""
    choice = (vg_color or "auto").lower()
    use = True
    if choice == "never":
        use = False
    elif choice == "always":
        use = True
    elif choice == "auto":
        if os.environ.get("NO_COLOR"):
            use = False
        elif os.environ.get("FORCE_COLOR") or os.environ.get("CLICOLOR_FORCE"):
            use = True
        elif sys.stdout.isatty():
            use = True
        else:
            use = False
    else:
        use = True  # unknown value -> on
    if use:
        os.environ["_VG_USE_COLOR"] = "1"
    else:
        os.environ.pop("_VG_USE_COLOR", None)
    return use


class C:
    """Color helper bound to a resolved on/off flag."""

    def __init__(self, use):
        self.use = use
        self.CYAN = "\033[36m" if use else ""
        self.GREEN = "\033[32m" if use else ""
        self.YELLOW = "\033[33m" if use else ""
        self.RED = "\033[31m" if use else ""
        self.BOLD = "\033[1m" if use else ""
        self.DIM = "\033[2m" if use else ""
        self.RESET = "\033[0m" if use else ""

    def cyan(self, s):
        return "%s%s%s" % (self.CYAN, s, self.RESET)

    def green(self, s):
        return "%s%s%s" % (self.GREEN, s, self.RESET)

    def yellow(self, s):
        return "%s%s%s" % (self.YELLOW, s, self.RESET)

    def red(self, s):
        return "%s%s%s" % (self.RED, s, self.RESET)

    def bold(self, s):
        return "%s%s%s" % (self.BOLD, s, self.RESET)

    def dim(self, s):
        return "%s%s%s" % (self.DIM, s, self.RESET)

    def error(self, s):
        sys.stderr.write("%s%s%s: %s\n" % (self.RED + self.BOLD, "error",
                                           self.RESET, s))

    def warn(self, s):
        sys.stderr.write("%s%s%s: %s\n" % (self.YELLOW + self.BOLD, "warning",
                                           self.RESET, s))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def init_registry():
    os.makedirs(MODELDIR, exist_ok=True)
    if not os.path.isfile(REGISTRY) or os.path.getsize(REGISTRY) == 0:
        with open(REGISTRY, "w") as f:
            f.write(DEFAULT_REGISTRY)


def read_registry():
    init_registry()
    out = {}
    with open(REGISTRY) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1]
    return out


def add_alias(alias, url):
    init_registry()
    rows = []
    found = False
    with open(REGISTRY) as f:
        for line in f:
            if line.strip().startswith(alias + " "):
                rows.append("%s %s\n" % (alias, url))
                found = True
            else:
                rows.append(line if line.endswith("\n") else line + "\n")
    if not found:
        rows.append("%s %s\n" % (alias, url))
    with open(REGISTRY, "w") as f:
        f.writelines(rows)


def model_path(alias):
    return os.path.join(MODELDIR, alias + ".gguf")


def model_installed(alias, c):
    if os.path.isfile(model_path(alias)):
        return True
    c.error("%s is not installed. Run: %s" % (c.cyan(alias),
                                              c.green("vgpu pull " + alias)))
    return False


def installed_models():
    res = {}
    for f in glob.glob(os.path.join(MODELDIR, "*.gguf")):
        a = os.path.basename(f)[:-5]
        res[a] = f
    return res


# ---------------------------------------------------------------------------
# URL / alias resolution
# ---------------------------------------------------------------------------
def resolve_url(arg):
    """Return (alias, url) or (None, None)."""
    if "huggingface.co" in arg:
        alias = os.path.basename(arg).replace(".gguf", "")
        return alias, arg
    if "/" in arg and arg.endswith(".gguf"):
        repo = arg.rsplit("/", 1)[0]
        file = arg.rsplit("/", 1)[1]
        alias = file.replace(".gguf", "")
        return alias, "https://huggingface.co/%s/resolve/main/%s" % (repo, file)
    reg = read_registry()
    if arg in reg:
        return arg, reg[arg]
    return None, None


# ---------------------------------------------------------------------------
# GPU reachability probe
# ---------------------------------------------------------------------------
def gpu_probe_build():
    if os.path.exists(GPU_PROBE_BIN) and os.access(GPU_PROBE_BIN, os.X_OK):
        return True
    cc = shutil.which("clang") or shutil.which("gcc")
    if not cc:
        return False
    inc = os.path.join(USRLIB, "..", "include")  # /usr/include
    if not os.path.isfile(os.path.join(inc, "vulkan", "vulkan.h")):
        inc = os.path.join(USRLIB, "include")
    if not os.path.isfile(os.path.join(inc, "vulkan", "vulkan.h")):
        return False
    try:
        subprocess.run([cc, GPU_PROBE_SRC, "-o", GPU_PROBE_BIN, "-I", inc,
                        "-L", USRLIB, "-lvulkan", "-rdynamic"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False)
        if os.path.exists(GPU_PROBE_BIN):
            os.chmod(GPU_PROBE_BIN, 0o755)
            return True
    except Exception:
        pass
    return False


def gpu_probe():
    """Run the probe; return (lines, rc). rc 0 => hardware GPU reachable."""
    if not gpu_probe_build():
        return "", 1
    if not os.access(GPU_PROBE_BIN, os.X_OK):
        return "", 1
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(
        p for p in [LIBDIR, os.path.join(LIBDIR, "vulkan"), USRLIB,
                    env.get("LD_LIBRARY_PATH", "")] if p)
    try:
        r = subprocess.run([GPU_PROBE_BIN], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           text=True)
        return r.stdout.strip(), r.returncode
    except Exception:
        return "", 1


def gpu_reachable():
    _, rc = gpu_probe()
    return rc == 0


# ---------------------------------------------------------------------------
# GGUF footprint + VRAM/OOM gates
# ---------------------------------------------------------------------------
def model_footprint(path, ctx):
    """Resident bytes estimate: weights (file size) + KV-cache."""
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return os.path.getsize(path)
            f.read(4)                      # version
            f.read(8)                      # tensor count
            kv = struct.unpack("<Q", f.read(8))[0]

            def rstr():
                ln = struct.unpack("<Q", f.read(8))[0]
                return f.read(ln).decode("utf-8", "replace")

            meta = {}

            def skip(t):
                if t == 8:
                    ln = struct.unpack("<Q", f.read(8))[0]; f.read(ln)
                elif t in (0, 1):
                    f.read(1)
                elif t in (2, 3):
                    f.read(2)
                elif t in (4, 5, 6):
                    f.read(4)
                elif t == 7:
                    f.read(1)
                elif t in (10, 11, 12):
                    f.read(8)
                elif t == 9:
                    et = struct.unpack("<I", f.read(4))[0]
                    c = struct.unpack("<Q", f.read(8))[0]
                    for _ in range(c):
                        skip(et)

            for _ in range(kv):
                key = rstr()
                t = struct.unpack("<I", f.read(4))[0]
                if t == 8:
                    ln = struct.unpack("<Q", f.read(8))[0]; meta[key] = f.read(ln)
                elif t == 4:
                    meta[key] = struct.unpack("<I", f.read(4))[0]
                elif t == 5:
                    meta[key] = struct.unpack("<i", f.read(4))[0]
                elif t == 10:
                    meta[key] = struct.unpack("<Q", f.read(8))[0]
                elif t == 11:
                    meta[key] = struct.unpack("<q", f.read(8))[0]
                elif t == 0:
                    meta[key] = struct.unpack("<B", f.read(1))[0]
                elif t == 1:
                    meta[key] = struct.unpack("<b", f.read(1))[0]
                elif t == 2:
                    meta[key] = struct.unpack("<H", f.read(2))[0]
                elif t == 3:
                    meta[key] = struct.unpack("<h", f.read(2))[0]
                elif t == 6:
                    meta[key] = struct.unpack("<f", f.read(4))[0]
                elif t == 7:
                    meta[key] = struct.unpack("<B", f.read(1))[0] != 0
                else:
                    skip(t)

            arch = meta.get("general.architecture", b"")
            arch = arch.decode() if isinstance(arch, bytes) else arch
            nl = meta.get(arch + ".block_count", 0) or 0
            ne = meta.get(arch + ".embedding_length", 0) or 0
            nh = meta.get(arch + ".attention.head_count", 0) or 0
            nkv = meta.get(arch + ".attention.head_count_kv", 0) or nh
            kvb = 0
            if nl and ne and nh:
                hd = ne // nh
                per_tok = nl * nkv * hd * 2          # K and V
                kvb = per_tok * ctx * 2              # f16 => 2 bytes/element
        return os.path.getsize(path) + kvb
    except Exception:
        try:
            return os.path.getsize(path)
        except Exception:
            return 0


def mem_available():
    """Bytes of reclaimable RAM available (MemAvailable, else MemFree).

    Returns None when it cannot be measured (so callers can skip the gate).
    """
    try:
        with open("/proc/meminfo") as f:
            avail = free = None
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    if parts[0] == "MemAvailable:":
                        avail = int(parts[1]) * 1024
                    elif parts[0] == "MemFree:":
                        free = int(parts[1]) * 1024
            if avail is not None:
                return avail
            if free is not None:
                return free
    except Exception:
        pass
    try:
        out = subprocess.check_output(["free", "-b"], text=True)
        for line in out.splitlines():
            if line.lower().startswith("mem:"):
                cols = line.split()
                if len(cols) >= 7:
                    return int(cols[6])
    except Exception:
        pass
    return None


def gate_oom(path, ctx, min_free_gb, no_oom_gate, c):
    """Refuse to launch if the model wouldn't fit. Returns True if OK, False if refused."""
    if no_oom_gate:
        return True
    avail = mem_available()
    if avail is None:
        return True
    try:
        fp = model_footprint(path, ctx)
    except Exception:
        return True
    if fp <= 0:
        return True
    minb = 0
    if min_free_gb:
        try:
            minb = int(float(min_free_gb) * 1024 ** 3)
        except (ValueError, TypeError):
            minb = 0
    need = fp + minb
    if need <= avail:
        return True
    fp_mib = round(fp / 1024 / 1024)
    avail_mib = round(avail / 1024 / 1024)
    need_mib = round(need / 1024 / 1024)
    c.error("%s would likely be OOM-killed: needs ~%dMiB (weights+KV@ctx=%d)"
            % (c.cyan(os.path.basename(path)), fp_mib, ctx))
    sys.stderr.write("%s  MemAvailable=%d MiB, required with --min-free %sGB margin = %d MiB.\n"
                     % (c.dim(""), avail_mib, min_free_gb or 0, need_mib))
    sys.stderr.write("%s  Refusing to launch to avoid an OOM kill. Use a smaller model, lower --vram/-c,\n"
                     % c.dim(""))
    sys.stderr.write("%s  set --min-free to reserve less headroom, or pass --no-oom-gate to override.\n"
                     % c.dim(""))
    return False


def gate_ctx(path, ctx, vram_gb, c):
    """Return a context size that fits the VRAM budget, or None to refuse."""
    if not vram_gb:
        return ctx
    try:
        budget = int(float(vram_gb) * 1024 ** 3)
    except ValueError:
        return ctx
    if model_footprint(path, ctx) <= budget:
        return ctx
    for cc in (2048, 1024, 512):
        if model_footprint(path, cc) <= budget:
            c.warn("VRAM budget %sGB: context auto-reduced %d -> %d to fit."
                   % (vram_gb, ctx, cc))
            return cc
    try:
        mib = round(os.path.getsize(path) / 1024 / 1024)
    except Exception:
        mib = 0
    c.error("%s needs ~%dMiB (weights alone) which exceeds --vram %sGB. "
            "Use a smaller model or raise --vram." %
            (c.cyan(os.path.basename(path)), mib, vram_gb))
    return None


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------
def server_alive():
    if not os.path.isfile(PIDFILE):
        return False
    try:
        with open(PIDFILE) as f:
            pid = int(f.read().strip())
    except Exception:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_port():
    try:
        with open(PORTFILE) as f:
            return int(f.read().strip())
    except Exception:
        return None


def server_is_gateway(port=None):
    """True if the running server is the multi-model gateway (serves /vgpu/status)."""
    if port is None:
        port = read_port() or DEFAULT_PORT
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:%d/vgpu/status" % port,
                                    timeout=2) as r:
            data = r.read()
        import json
        d = json.loads(data)
        return "backend" in d
    except Exception:
        return False


def server_model_get():
    try:
        with open(MODELFILE) as f:
            return f.read().strip()
    except Exception:
        return ""


def server_model_set(alias):
    try:
        with open(MODELFILE, "w") as f:
            f.write(str(alias))
    except Exception:
        pass


def server_model_clear():
    try:
        os.remove(MODELFILE)
    except OSError:
        pass


def server_backend():
    """Best-effort detection of the compute backend a running server uses.

    Returns one of: "gpu", "cpu", or None (unknown / no server / can't tell).
    We inspect the live process's memory map for the Vulkan ggml backend
    library (libggml-vulkan.so) and the system Vulkan driver -- the same
    signal used to confirm GPU inference is actually active, rather than
    trusting the launch flags (which lie after an auto-fallback to CPU).
    """
    if not server_alive():
        return None
    try:
        pid = int(open(PIDFILE).read().strip())
    except Exception:
        return None
    maps = "/proc/%d/maps" % pid
    try:
        with open(maps, "r", errors="replace") as f:
            for line in f:
                if "libggml-vulkan.so" in line:
                    return "gpu"
                # Some builds split the backend into a vulkan/ subdir.
                if "/vulkan/" in line and "ggml" in line:
                    return "gpu"
    except OSError:
        return None
    return "cpu"


# ---------------------------------------------------------------------------
# Backend launcher (with CPU auto-fallback)
# ---------------------------------------------------------------------------
def start_backend(model, port, log, force_cpu, ctx=DEFAULT_CTX):
    """Launch llama-server (detached). Returns the child pid."""
    open(log, "w").close()
    env = dict(os.environ)
    # Force Vulkan ON (a stray OLLAMA_VULKAN=0 would silently push to CPU).
    env["OLLAMA_VULKAN"] = "1"
    env["LD_LIBRARY_PATH"] = ":".join(
        p for p in [LIBDIR, os.path.join(LIBDIR, "vulkan"), USRLIB,
                    env.get("LD_LIBRARY_PATH", "")] if p)
    if force_cpu:
        env.pop("GGML_BACKEND_PATH", None)
        env.pop("GGML_VK_VISIBLE_DEVICES", None)
    else:
        env["GGML_BACKEND_PATH"] = os.path.join(LIBDIR, "vulkan",
                                                "libggml-vulkan.so")
        env["GGML_VK_VISIBLE_DEVICES"] = "0"
    ngl = 0 if force_cpu else 99
    args = [os.path.join(LIBDIR, "llama-server"), "-m", model, "-ngl", str(ngl),
            "-fa", "off", "-c", str(ctx), "--host", "0.0.0.0", "--port",
            str(port), "--log-verbosity", "3"]
    # start_new_session=True detaches the backend into its own session/process
    # group (equivalent to `setsid`), so it survives the CLI exiting.
    pid = subprocess.Popen(args, env=env, stdout=open(log, "a"),
                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                           start_new_session=True).pid
    with open(PIDFILE, "w") as f:
        f.write(str(pid))
    with open(PORTFILE, "w") as f:
        f.write(str(port))
    return pid


def wait_ready(log, pid, tries=90, delay=1.0):
    import time
    for _ in range(tries):
        try:
            with open(log) as f:
                content = f.read()
        except Exception:
            content = ""
        if "listening on" in content:
            return True
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        if ("error loading model" in content.lower()
                or "exiting due to model loading error" in content.lower()):
            return False
        time.sleep(delay)
    return False


def start_with_fallback(model, port, log, force_cpu, alias, c, ctx=DEFAULT_CTX):
    """Start the backend, auto-falling back to CPU on GPU failure.

    Prints a warning when falling back. Returns mode ("gpu"/"cpu") or None.
    """
    if force_cpu:
        sys.stderr.write(
            "%sWARNING: running on CPU only (--cpu). This is MUCH slower than "
            "the Vulkan GPU.\n        Use CPU only for quants Vulkan-broken on "
            "this PowerVR device:\n          - K-quants crash the driver "
            "(createComputePipeline ErrorUnknown)\n          - IQ3_XS silently "
            "corrupts output\n        Q4_0 / Q8_0 / IQ4_XS run fine on the GPU.\n"
            % c.YELLOW)
        pid = start_backend(model, port, log, 1, ctx)
        if wait_ready(log, pid):
            sys.stderr.write("%s>>> running on %s <<<\n"
                             % (c.YELLOW, c.bold("CPU") + c.YELLOW))
            return "cpu"
        c.error("CPU server failed to start:")
        _tail(log, 15)
        return None

    sys.stderr.write("%s %s server for %s on port %s ...\n"
                     % (c.bold("Starting"), c.cyan("Vulkan GPU"),
                        c.cyan(alias), c.cyan(str(port))))
    pid = start_backend(model, port, log, 0, ctx)
    if wait_ready(log, pid):
        sys.stderr.write("%s>>> running on %s <<<\n"
                         % (c.green(""), c.bold("GPU (Vulkan)")))
        return "gpu"
    # GPU failed -> auto-fallback to CPU with a prominent warning.
    try:
        os.kill(pid, 15)
    except OSError:
        pass
    import time
    time.sleep(1)
    sys.stderr.write(
        "\n%s!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        "!! WARNING: Vulkan GPU backend failed to start (likely a K-quant    !!\n"
        "!! driver crash: createComputePipeline ErrorUnknown).              !!\n"
        "!! Auto-falling back to CPU. This is MUCH SLOWER.                  !!\n"
        "!! NOTE: silently-corrupting quants (e.g. IQ3_XS) do NOT crash and  !!\n"
        "!!   will still run on GPU -- use 'vgpu run --cpu <alias>' for those. !!\n"
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%s\n\n"
        % (c.YELLOW, c.RESET))
    pid = start_backend(model, port, log, 1, ctx)
    if wait_ready(log, pid):
        sys.stderr.write("%s>>> running on %s <<<\n"
                         % (c.YELLOW, c.bold("CPU") + c.YELLOW))
        return "cpu"
    c.error("CPU fallback also failed to start:")
    _tail(log, 15)
    return None


def _tail(path, n=15):
    try:
        with open(path) as f:
            lines = f.read().splitlines()[-n:]
        for ln in lines:
            sys.stderr.write(ln + "\n")
    except Exception:
        pass


def stop_server(c):
    if not server_alive():
        sys.stderr.write(c.dim("no server running") + "\n")
        return
    try:
        with open(PIDFILE) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, 15)
        except OSError:
            pass
    except Exception:
        pass
    if os.path.isfile(BACKEND_PIDFILE):
        try:
            with open(BACKEND_PIDFILE) as f:
                bk = int(f.read().strip())
            try:
                os.kill(bk, 15)
            except OSError:
                pass
        except Exception:
            pass
    import time
    time.sleep(1)
    for p in (PIDFILE, BACKEND_PIDFILE, MODELFILE):
        try:
            os.remove(p)
        except OSError:
            pass
    sys.stderr.write(c.green("stopped") + "\n")
