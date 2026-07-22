#!/usr/bin/env python3
"""
vgpu_gateway.py - on-the-fly model loader/unloader in front of llama-server.

Listens on a PUBLIC host/port. Maintains a SINGLE llama-server backend on an
INTERNAL port. When a request names a model that is not the one currently
resident, the gateway stops the old backend and starts a new one for the
requested model (with VRAM gating + GPU->CPU auto-fallback). The previous
model's weights are freed on swap, so at most ONE model lives in RAM at a time
-- safe under a tight VRAM/RAM budget on integrated GPUs.

Endpoints:
  POST /v1/chat/completions   auto-swaps to the requested model (stream + non-stream)
  POST /v1/completions        auto-swaps to the requested model
  POST /v1/embeddings         auto-swaps to the requested model
  GET  /v1/models             list installed models (+ which is currently loaded)
  POST /v1/unload             unload the named model (mirrors OpenAI /v1/unload)
  POST /vgpu/unload           explicitly free the currently-loaded model
  GET  /vgpu/status           JSON status (current model, mode, ports, pids)
  GET  /healthz               200 when the gateway process is up

  Ollama-native compatibility (so Ollama clients just work):
  POST /api/chat              -> /v1/chat/completions (NDJSON, Ollama-shaped)
  POST /api/generate          -> /v1/completions (NDJSON, Ollama-shaped)
  GET  /api/tags              list local models (Ollama schema)
  GET  /api/ps                list loaded models (Ollama schema)
  POST /api/embed             embeddings (input=text|list)
  POST /api/embeddings        legacy embeddings (prompt=text)
  POST /api/show              model info
  POST /api/delete            delete a local model
  POST /api/copy              copy a model (alias an existing GGUF)
  POST /api/create            alias an existing GGUF (no Modelfile support)
  POST /api/pull              delegate to `vgpu pull`
  GET  /api/version           vgpu version string

  keep_alive (Ollama semantics, passed via the OpenAI request body): controls how long
  the just-used model stays resident. "0" / 0 -> unload immediately after the reply;
  any negative number / "-1" / "-1m" -> keep loaded forever; a positive duration string
  ("5m", "30s") or seconds -> unload after that idle period. Default: 5m. An empty
  prompt also just loads (warms) the model without generating.

Launched by `vgpu gateway`. Config comes from CLI flags.
"""
import os, sys, json, struct, time, signal, glob, subprocess, threading, argparse, re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import HTTPConnection

HOME = os.environ.get("HOME", "/data/data/com.termux/files/home")
MODEL_DIR = os.path.join(HOME, "vgpu_models")
LIBDIR = "/data/data/com.termux/files/usr/lib/ollama"
USRLIB = "/data/data/com.termux/files/usr/lib"
DEFAULT_CTX = 4096
DEFAULT_KEEP_ALIVE = 300   # seconds (5m), mirrors Ollama's default


# ---------------------------------------------------------------------------
# GGUF footprint + VRAM gate (ported from vgpu's bash model_footprint/gate_vram)
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

            def rstr(f):
                ln = struct.unpack("<Q", f.read(8))[0]
                return f.read(ln).decode("utf-8", "replace")

            meta = {}

            def skip(f, t):
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
                        skip(f, et)

            for _ in range(kv):
                key = rstr(f)
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
                    skip(f, t)

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
    On Android/Termux MemAvailable already accounts for reclaimable caches and
    is the right number to compare a model's footprint against.
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
    # last resort: coreutils `free -b` (column 7 = available on Linux)
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


def gate_oom(path, ctx, min_free_gb, no_oom_gate):
    """Refuse to launch if the model's footprint would not fit in available RAM.

    Returns None to allow launch, or a human-readable refusal message (str).
    no_oom_gate=True or unmeasurable memory => allow (None).
    """
    if no_oom_gate:
        return None
    avail = mem_available()
    if avail is None:
        return None
    try:
        fp = model_footprint(path, ctx)
    except Exception:
        return None
    minb = 0
    if min_free_gb:
        try:
            minb = int(float(min_free_gb) * 1024 ** 3)
        except (ValueError, TypeError):
            minb = 0
    need = fp + minb
    if need <= avail:
        return None
    fp_mib = round(fp / 1024 / 1024)
    avail_mib = round(avail / 1024 / 1024)
    need_mib = round(need / 1024 / 1024)
    return ("Model %s would likely be OOM-killed: needs ~%dMiB "
            "(weights+KV@ctx=%d), but only %dMiB is available (with --min-free %sGB "
            "margin that becomes %dMiB). Use a smaller model, a lower --vram/-c, a "
            "smaller --min-free, or pass --no-oom-gate to override."
            % (os.path.basename(path), fp_mib, ctx, avail_mib, min_free_gb or 0, need_mib))


def gate_ctx(path, ctx, vram_gb):
    """Return a context size that fits the VRAM budget, or None to refuse."""
    if not vram_gb:
        return ctx
    try:
        budget = int(float(vram_gb) * 1024 ** 3)
    except ValueError:
        return ctx
    if model_footprint(path, ctx) <= budget:
        return ctx
    for c in (2048, 1024, 512):
        if model_footprint(path, c) <= budget:
            return c
    return None


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
def installed_models():
    res = {}
    for f in glob.glob(os.path.join(MODEL_DIR, "*.gguf")):
        a = os.path.basename(f)[:-5]
        res[a] = f
    return res


def resolve_model(name, models):
    """Map a client-supplied model id (alias, basename, or path) to (alias, path)."""
    if name in models:
        return name, models[name]
    cand = name if os.path.isabs(name) else os.path.join(MODEL_DIR, name + ".gguf")
    if os.path.exists(cand):
        return os.path.basename(cand)[:-5], cand
    # also accept a bare path that already ends in .gguf
    if os.path.exists(name) and name.endswith(".gguf"):
        return os.path.basename(name)[:-5], name
    return None, None


# ---------------------------------------------------------------------------
# Gateway: owns the single backend process
# ---------------------------------------------------------------------------
class Gateway:
    def __init__(self, backend_host, backend_port, public_host, public_port,
                 vram, force_cpu, default_model, keep_alive=None,
                 min_free=None, no_oom_gate=False):
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.public_host = public_host
        self.public_port = public_port
        self.vram = vram
        self.force_cpu = force_cpu
        self.default_model = default_model
        self.min_free = min_free
        self.no_oom_gate = no_oom_gate
        self.current = None          # alias currently resident
        self.mode = None             # "gpu" | "cpu"
        self.backend_pid = None
        self.lock = threading.Lock()
        self.backend_log = os.path.join(MODEL_DIR, "gateway.backend.log")
        self.backend_pidfile = os.path.join(MODEL_DIR, ".gateway.backend.pid")
        # keep_alive: seconds the current model stays resident after a request.
        # None => use the server default. A threading.Timer unloads it when idle.
        self.keep_alive = None
        self._unload_timer = None
        self._server_default_keep_alive = (self.parse_keep_alive(keep_alive)
                                           if keep_alive else DEFAULT_KEEP_ALIVE)

    # -- lifecycle ---------------------------------------------------------
    def backend_alive(self):
        if self.backend_pid is None:
            return False
        try:
            os.kill(self.backend_pid, 0)
        except OSError:
            return False
        return True

    def _spawn(self, alias, path, ctx, force_cpu):
        open(self.backend_log, "w").close()
        env = os.environ.copy()
        env["OLLAMA_VULKAN"] = "1"
        env["LD_LIBRARY_PATH"] = "%s:%s/vulkan:%s" % (LIBDIR, LIBDIR, USRLIB)
        if force_cpu:
            env.pop("GGML_BACKEND_PATH", None)
            env.pop("GGML_VK_VISIBLE_DEVICES", None)
        else:
            env["GGML_BACKEND_PATH"] = "%s/vulkan/libggml-vulkan.so" % LIBDIR
            env["GGML_VK_VISIBLE_DEVICES"] = "0"
        ngl = "0" if force_cpu else "99"
        args = [os.path.join(LIBDIR, "llama-server"), "-m", path, "-ngl", ngl,
                "-fa", "off", "-c", str(ctx),
                "--host", self.backend_host, "--port", str(self.backend_port),
                "--log-verbosity", "3", "--embeddings"]
        p = subprocess.Popen(args, env=env,
                             stdout=open(self.backend_log, "a"),
                             stderr=subprocess.STDOUT)
        self.backend_pid = p.pid
        with open(self.backend_pidfile, "w") as fh:
            fh.write(str(p.pid))

    def _wait_ready(self, timeout=90):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.backend_alive():
                return False
            try:
                c = HTTPConnection(self.backend_host, self.backend_port, timeout=2)
                c.request("GET", "/v1/models")
                r = c.getresponse()
                r.read()
                c.close()
                if r.status == 200:
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        return False

    def _kill(self):
        if self.backend_pid:
            try:
                os.kill(self.backend_pid, signal.SIGTERM)
            except OSError:
                pass
            for _ in range(25):
                if not self.backend_alive():
                    break
                time.sleep(0.2)
            else:
                try:
                    os.kill(self.backend_pid, signal.SIGKILL)
                except OSError:
                    pass
            self.backend_pid = None
        try:
            os.remove(self.backend_pidfile)
        except OSError:
            pass

    def _start(self, alias, path):
        """Launch backend (GPU, then CPU fallback). Returns (mode, err)."""
        # OOM gate: refuse to launch if the model wouldn't fit in free RAM,
        # so the OOM killer can't silently kill the backend (and the gateway).
        oom_err = gate_oom(path, DEFAULT_CTX, self.min_free, self.no_oom_gate)
        if oom_err:
            return None, oom_err
        ctx = gate_ctx(path, DEFAULT_CTX, self.vram)
        if ctx is None:
            return None, ("Model %s cannot fit --vram %sGB (weights alone exceed the "
                          "budget). Use a smaller model or raise --vram." % (alias, self.vram))
        self._spawn(alias, path, ctx, force_cpu=self.force_cpu)
        if self._wait_ready():
            self.mode = "cpu" if self.force_cpu else "gpu"
            sys.stderr.write("[vgpu] %s for %s on %s:%s\n"
                             % (("running on CPU" if self.force_cpu
                                 else "running on GPU (Vulkan)"),
                                alias, self.backend_host, self.backend_port))
            sys.stderr.flush()
            return self.mode, None
        if self.force_cpu:
            return None, "Backend failed to start (CPU). See %s" % self.backend_log
        # GPU launch failed -> auto-fallback to CPU
        self._kill()
        self._spawn(alias, path, ctx, force_cpu=1)
        if self._wait_ready():
            self.mode = "cpu"
            sys.stderr.write("[vgpu] running on CPU (Vulkan GPU failed, fell "
                             "back) for %s on %s:%s\n"
                             % (alias, self.backend_host, self.backend_port))
            sys.stderr.flush()
            return "cpu", None
        return None, "Backend failed to start (GPU+CPU fallback). See %s" % self.backend_log

    def ensure_model(self, alias):
        """Load `alias` if not already resident. Returns (mode, err)."""
        models = installed_models()
        a, p = resolve_model(alias, models)
        if not p:
            return None, "Unknown model '%s'. Available: %s" % (
                alias, ", ".join(sorted(models)) or "(none)")
        if self.current == a and self.backend_alive():
            return self.mode, None
        # swap: unload previous, then load requested
        self._kill()
        self.current = None
        self.mode = None
        mode, err = self._start(a, p)
        if err:
            return None, err
        self.current = a
        return mode, None

    def stop_backend(self):
        self._kill()
        self.current = None
        self.mode = None

    @staticmethod
    def parse_keep_alive(v):
        """Ollama semantics -> seconds. '0'/0 => unload now (0s);
        negative/'-1'/'-1m' => keep forever (None); '5m'/'30s'/number => seconds."""
        if v is None:
            return DEFAULT_KEEP_ALIVE
        s = str(v).strip().lower()
        try:
            n = float(s)
            if n < 0:
                return None
            return 0 if n == 0 else int(n)
        except ValueError:
            pass
        m = re.match(r"(-?\d+(?:\.\d+)?)\s*([smh])?", s)
        if m:
            val = float(m.group(1))
            unit = m.group(2)
            secs = val * (60 if unit == "m" else 3600 if unit == "h" else 1)
            if val < 0:
                return None
            return 0 if secs == 0 else int(secs)
        return DEFAULT_KEEP_ALIVE

    def _cancel_unload(self):
        if self._unload_timer is not None:
            self._unload_timer.cancel()
            self._unload_timer = None

    def _schedule_unload(self, seconds):
        self._cancel_unload()
        if seconds is None:        # keep forever
            return
        if seconds <= 0:           # unload immediately
            self.stop_backend()
            return
        self._unload_timer = threading.Timer(seconds, self.stop_backend)
        self._unload_timer.daemon = True
        self._unload_timer.start()

    def status(self):
        with self.lock:
            ka = self.keep_alive
            if ka is None:
                ka = self._server_default_keep_alive
            return {
                "public": "%s:%s" % (self.public_host, self.public_port),
                "backend": "%s:%s" % (self.backend_host, self.backend_port),
                "current_model": self.current,
                "mode": self.mode,
                "backend_pid": self.backend_pid,
                "vram_gb": self.vram,
                "keep_alive_secs": ka,
                "models": sorted(installed_models()),
            }


# ---------------------------------------------------------------------------
# Ollama-native API compatibility layer (/api/*)
# ---------------------------------------------------------------------------
# The backend (llama-server) already speaks the OpenAI-compatible API. Ollama's
# native endpoints use different request/response shapes, so we translate here.
# Supported: /api/chat, /api/generate, /api/tags, /api/ps, /api/embed,
#            /api/embeddings, /api/show, /api/version, /api/create, /api/delete,
#            /api/copy, /api/pull (pull delegates to `vgpu pull`).
# NOTE: /api/chat & /api/generate stream newline-delimited JSON objects (NDJSON),
# not SSE; we reshape the backend's SSE on the fly.
VGPU_VERSION = "0.1.0"

_OLLAMA_DETAILS_FAMILY = {
    "qwen2": "qwen2", "qwen3": "qwen3", "llama": "llama",
    "mistral": "mistral", "gemma": "gemma", "phi": "phi",
}


def _ollama_family(alias):
    a = alias.lower()
    for k, v in _OLLAMA_DETAILS_FAMILY.items():
        if a.startswith(k):
            return v
    return "unknown"


def ollama_chat_request(body):
    """Translate an /api/chat body into an OpenAI /v1/chat/completions body."""
    out = {"model": body.get("model", ""),
           "messages": body.get("messages", []),
           "stream": body.get("stream", True)}
    if "format" in body:
        out["response_format"] = {"type": "json_object"} if body["format"] == "json" \
            else {"type": "json_schema", "json_schema": {"schema": body["format"]}}
    if "keep_alive" in body:
        out["keep_alive"] = body["keep_alive"]
    _ollama_options(out, body)
    return out


def ollama_generate_request(body):
    """Translate an /api/generate body into an OpenAI /v1/completions body."""
    out = {"model": body.get("model", ""),
           "prompt": body.get("prompt", ""),
           "stream": body.get("stream", True)}
    if "format" in body:
        out["response_format"] = {"type": "json_object"} if body["format"] == "json" \
            else {"type": "json_schema", "json_schema": {"schema": body["format"]}}
    if "keep_alive" in body:
        out["keep_alive"] = body["keep_alive"]
    _ollama_options(out, body)
    return out


def _ollama_options(out, body):
    """Map Ollama 'options' (sampler params) onto OpenAI request fields."""
    opts = body.get("options") or {}
    if not isinstance(opts, dict):
        return
    if "num_predict" in opts:
        out["max_tokens"] = int(opts["num_predict"])
    if "temperature" in opts:
        out["temperature"] = float(opts["temperature"])
    if "top_p" in opts:
        out["top_p"] = float(opts["top_p"])
    if "top_k" in opts:
        out["top_k"] = int(opts["top_k"])
    if "seed" in opts:
        out["seed"] = int(opts["seed"])
    if "stop" in opts:
        out["stop"] = opts["stop"]


def ollama_tags_response(status):
    """Reshape our /v1/models-style status into Ollama's GET /api/tags."""
    models = []
    installed = {m: os.path.join(MODEL_DIR, m + ".gguf") for m in status.get("models", [])}
    cur = status.get("current_model")
    for a in sorted(installed):
        models.append({
            "name": a,
            "model": a,
            "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z",
                                         time.localtime(os.path.getmtime(installed[a]))),
            "size": os.path.getsize(installed[a]),
            "digest": "vgpu-" + a,
            "details": {
                "format": "gguf",
                "family": _ollama_family(a),
                "families": [_ollama_family(a)],
                "parameter_size": _param_label(a),
                "quantization_level": _quant_label(a),
            },
        })
    return {"models": models}


def ollama_ps_response(status):
    """Reshape status into Ollama's GET /api/ps."""
    cur = status.get("current_model")
    models = []
    if cur:
        path = os.path.join(MODEL_DIR, cur + ".gguf")
        import datetime
        ka = status.get("keep_alive_secs")
        expires = ""
        if ka:
            expires = (datetime.datetime.now() +
                       datetime.timedelta(seconds=ka)).strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        models.append({
            "name": cur, "model": cur,
            "size": os.path.getsize(path) if os.path.exists(path) else 0,
            "digest": "vgpu-" + cur,
            "details": {
                "format": "gguf", "family": _ollama_family(cur),
                "families": [_ollama_family(cur)],
                "parameter_size": _param_label(cur),
                "quantization_level": _quant_label(cur),
            },
            "expires_at": expires,
            "size_vram": 0,
        })
    return {"models": models}


def _param_label(alias):
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", alias)
    return (m.group(1) + "B") if m else "unknown"


def _quant_label(alias):
    m = re.search(r"(q\d+[_-]?[a-z\d]*)", alias, re.I)
    return m.group(1).upper().replace("-", "_") if m else "unknown"


def ollama_show_response(alias):
    """Mirror Ollama's POST /api/show response (what ollama_dart's ShowResponse
    deserializes). Keep every field present so the client never 500s on a
    missing key; capabilities use Ollama's canonical string values."""
    path = os.path.join(MODEL_DIR, alias + ".gguf")
    if not os.path.exists(path):
        return None
    modified = time.strftime("%Y-%m-%dT%H:%M:%S%z",
                              time.localtime(os.path.getmtime(path)))
    details = {
        "parent_model": "",
        "format": "gguf",
        "family": _ollama_family(alias),
        "families": [_ollama_family(alias)],
        "parameter_size": _param_label(alias),
        "quantization_level": _quant_label(alias),
    }
    return {
        "modelfile": "FROM %s" % path,
        "parameters": "",
        "template": "",
        "system": "",
        "details": details,
        "model_info": {},
        "projector_info": {},
        "modified_at": modified,
        "capabilities": ["completion", "tools"],
    }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"   # close-per-response => streaming needs no chunking

    def log_message(self, *a):
        pass

    def _read_body(self):
        cl = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(cl) if cl > 0 else b""

    def _send(self, status, ctype, data):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        # CORS: mirror Ollama so browser-based / OpenAI-compatible clients can
        # talk to the gateway (e.g. when accessed from a web context).
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods",
                         "GET, POST, PUT, DELETE, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers",
                         "Authorization, Content-Type, User-Agent, Accept, "
                         "X-Requested-With, OpenAI-Beta, x-stainless-lang, "
                         "x-stainless-os, x-stainless-package-version, "
                         "x-stainless-runtime, x-stainless-runtime-version")
        self.end_headers()
        try:
            self.wfile.write(data)
        except Exception:
            pass

    def _send_plain(self, status, text):
        self._send(status, "text/plain; charset=utf-8", text.encode("utf-8"))

    def do_HEAD(self):
        # Mirror Ollama: HEAD / -> "Ollama is running"; HEAD /api/version -> JSON.
        gw = self.server.gw
        if self.path in ("/", "/healthz"):
            self._send_plain(200, "Ollama is running")
        elif self.path == "/api/version":
            self._send(200, "application/json",
                       json.dumps({"version": VGPU_VERSION}).encode())
        elif self.path == "/api/tags":
            self._send(200, "application/json", b"{}")
        else:
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

    def do_OPTIONS(self):
        # CORS preflight.
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods",
                         "GET, POST, PUT, DELETE, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers",
                         "Authorization, Content-Type, User-Agent, Accept, "
                         "X-Requested-With, OpenAI-Beta, x-stainless-lang, "
                         "x-stainless-os, x-stainless-package-version, "
                         "x-stainless-runtime, x-stainless-runtime-version")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _forward(self, method, path, headers, body):
        gw = self.server.gw
        try:
            c = HTTPConnection(gw.backend_host, gw.backend_port, timeout=600)
            fwd = {}
            for k, v in headers.items():
                if k.lower() in ("host", "content-length", "connection",
                                 "transfer-encoding", "accept-encoding"):
                    continue
                fwd[k] = v
            fwd["Host"] = "%s:%s" % (gw.backend_host, gw.backend_port)
            if body:
                fwd["Content-Length"] = str(len(body))
            else:
                fwd["Content-Length"] = "0"
            c.request(method, path, body=body or None, headers=fwd)
            r = c.getresponse()
            self.send_response(r.status)
            for k, v in r.getheaders():
                if k.lower() in ("transfer-encoding", "connection",
                                 "content-length", "content-encoding"):
                    continue
                self.send_header(k, v)
            self.end_headers()
            while True:
                chunk = r.read(65536)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except Exception:
                    break
            c.close()
        except Exception as e:
            try:
                self._send(502, "application/json",
                           json.dumps({"error": "backend proxy error: %s" % e}).encode())
            except Exception:
                pass

    def do_GET(self):
        gw = self.server.gw
        if self.path == "/v1/models":
            self._models()
        elif self.path == "/vgpu/status":
            self._send(200, "application/json",
                       json.dumps(gw.status()).encode())
        elif self.path == "/":
            # Faithful to Ollama: GET / returns exactly "Ollama is running"
            # (status 200). The ollama-app client validates the host by GET-ing
            # "/" and requiring this exact body, so it MUST match.
            self._send_plain(200, "Ollama is running")
        elif self.path == "/healthz":
            self._send_plain(200, "Ollama is running")
        # ---- Ollama-native GET endpoints ----
        elif self.path == "/api/tags":
            self._send(200, "application/json",
                       json.dumps(ollama_tags_response(gw.status())).encode())
        elif self.path == "/api/ps":
            self._send(200, "application/json",
                       json.dumps(ollama_ps_response(gw.status())).encode())
        elif self.path == "/api/version":
            self._send(200, "application/json",
                       json.dumps({"version": VGPU_VERSION}).encode())
        elif self.path == "/api/show":
            self._send(501, "application/json",
                       json.dumps({"error": "use POST /api/show with a model body"}).encode())
        else:
            with gw.lock:
                if not gw.backend_alive():
                    self._send(503, "application/json",
                               json.dumps({"error": "no model loaded; POST to "
                                           "/v1/chat/completions with a 'model' first"}).encode())
                    return
                self._forward("GET", self.path, self.headers, b"")

    def do_POST(self):
        gw = self.server.gw
        body = self._read_body()
        if self.path == "/v1/unload":
            self._handle_unload(body)
            return
        if self.path == "/vgpu/unload":
            with gw.lock:
                gw.stop_backend()
            self._send(200, "application/json", json.dumps({"unloaded": True}).encode())
            return
        if self.path in ("/v1/chat/completions", "/v1/completions", "/v1/embeddings"):
            self._model_request(body)
            return
        # ---- Ollama-native POST endpoints ----
        if self.path == "/api/chat":
            try:
                req = json.loads(body) if body else {}
            except Exception:
                self._send(400, "application/json", json.dumps({"error": "invalid JSON"}).encode())
                return
            self._ollama_infer("/v1/chat/completions", ollama_chat_request(req))
            return
        if self.path == "/api/generate":
            try:
                req = json.loads(body) if body else {}
            except Exception:
                self._send(400, "application/json", json.dumps({"error": "invalid JSON"}).encode())
                return
            self._ollama_infer("/v1/completions", ollama_generate_request(req))
            return
        if self.path == "/api/embed" or self.path == "/api/embeddings":
            self._ollama_embed_post(body)
            return
        if self.path == "/api/show":
            self._ollama_show(body)
            return
        if self.path == "/api/delete":
            self._ollama_delete(body)
            return
        if self.path == "/api/copy":
            self._ollama_copy(body)
            return
        if self.path == "/api/create":
            # vgpu has no Modelfile concept; create = register an alias to a local GGUF.
            self._ollama_create(body)
            return
        if self.path == "/api/pull":
            self._ollama_pull(body)
            return
        # anything else: proxy to backend if a model is loaded
        with gw.lock:
            if not gw.backend_alive():
                self._send(503, "application/json",
                           json.dumps({"error": "no model loaded"}).encode())
                return
            self._forward("POST", self.path, self.headers, body)

    def _handle_unload(self, body):
        gw = self.server.gw
        try:
            req = json.loads(body) if body else {}
        except Exception:
            req = {}
        target = req.get("model")
        with gw.lock:
            # If a specific model is named, only unload when it is the resident one.
            if target and gw.current and target != gw.current:
                self._send(200, "application/json",
                           json.dumps({"unloaded": False,
                                       "current_model": gw.current,
                                       "note": "requested model is not the one loaded"}).encode())
                return
            gw.stop_backend()
        self._send(200, "application/json", json.dumps({"unloaded": True}).encode())

    def _model_request(self, body):
        gw = self.server.gw
        try:
            req = json.loads(body) if body else {}
        except Exception:
            self._send(400, "application/json",
                       json.dumps({"error": "invalid JSON body"}).encode())
            return
        alias = req.get("model") or gw.default_model
        if not alias:
            self._send(400, "application/json",
                       json.dumps({"error": "no 'model' specified and no default set"}).encode())
            return
        # Warm-only request (empty prompt / messages): load but don't generate.
        msgs = req.get("messages") if self.path == "/v1/chat/completions" else (
            req.get("prompt") if self.path == "/v1/completions" else req.get("input"))
        is_warmup = (not msgs) or (isinstance(msgs, (list, str)) and len(msgs) == 0)
        with gw.lock:
            mode, err = gw.ensure_model(alias)
            if err:
                self._send(503, "application/json", json.dumps({"error": err}).encode())
                return
            # Parse keep_alive (per-request overrides server default).
            ka = req.get("keep_alive")
            ka_secs = gw.parse_keep_alive(ka) if ka is not None else gw._server_default_keep_alive
            if is_warmup:
                # Just loaded the model; schedule its idle unload and report.
                gw._schedule_unload(ka_secs)
                self._send(200, "application/json",
                           json.dumps({"loaded": True, "model": gw.current, "mode": gw.mode,
                                       "keep_alive": ka_secs}).encode())
                return
            # forward under the lock: serializes swaps vs. other requests so a
            # swap can never leave an in-flight request served by the wrong model
            self._forward(self.command, self.path, self.headers, body)
            gw._schedule_unload(ka_secs)

    # -- Ollama-native POST helpers ---------------------------------------
    def _ollama_embed_post(self, body):
        gw = self.server.gw
        try:
            req = json.loads(body) if body else {}
        except Exception:
            self._send(400, "application/json", json.dumps({"error": "invalid JSON"}).encode())
            return
        alias = req.get("model") or gw.default_model
        if not alias:
            self._send(400, "application/json",
                       json.dumps({"error": "no 'model' specified"}).encode())
            return
        # /api/embed uses 'input'; legacy /api/embeddings uses 'prompt'.
        inp = req.get("input", req.get("prompt", ""))
        with gw.lock:
            mode, err = gw.ensure_model(alias)
            if err:
                self._send(503, "application/json", json.dumps({"error": err}).encode())
                return
            ka = req.get("keep_alive")
            ka_secs = gw.parse_keep_alive(ka) if ka is not None else gw._server_default_keep_alive
        try:
            c = HTTPConnection(gw.backend_host, gw.backend_port, timeout=120)
            c.request("POST", "/v1/embeddings",
                      body=json.dumps({"model": gw.current, "input": inp}).encode(),
                      headers={"Content-Type": "application/json"})
            r = c.getresponse()
            data = r.read()
            c.close()
            if r.status != 200:
                gw._schedule_unload(ka_secs)
                self._send(r.status, "application/json", data)
                return
            o = json.loads(data) if data else {}
            embs = [e.get("embedding") for e in o.get("data", [])] or o.get("embeddings", [])
            if isinstance(inp, list):
                out = {"model": gw.current, "embeddings": embs}
            else:
                out = {"model": gw.current,
                       "embedding": embs[0] if embs else []}
        except Exception as e:
            self._send(502, "application/json",
                       json.dumps({"error": "backend proxy error: %s" % e}).encode())
            return
        gw._schedule_unload(ka_secs)
        self._send(200, "application/json", json.dumps(out).encode())

    def _ollama_show(self, body):
        try:
            req = json.loads(body) if body else {}
        except Exception:
            req = {}
        alias = req.get("model")
        if not alias:
            self._send(400, "application/json", json.dumps({"error": "no 'model' specified"}).encode())
            return
        a, _ = resolve_model(alias, installed_models())
        if not a:
            self._send(404, "application/json",
                       json.dumps({"error": "model '%s' not found" % alias}).encode())
            return
        res = ollama_show_response(a)
        self._send(200, "application/json", json.dumps(res).encode())

    def _ollama_delete(self, body):
        try:
            req = json.loads(body) if body else {}
        except Exception:
            req = {}
        alias = (req.get("model") or "").strip()
        if not alias:
            self._send(400, "application/json", json.dumps({"error": "no 'model' specified"}).encode())
            return
        a, p = resolve_model(alias, installed_models())
        if not p:
            self._send(404, "application/json",
                       json.dumps({"error": "model '%s' not found" % alias}).encode())
            return
        try:
            os.remove(p)
        except OSError as e:
            self._send(500, "application/json", json.dumps({"error": str(e)}).encode())
            return
        self._send(200, "text/plain", b"deleted\n")

    def _ollama_copy(self, body):
        try:
            req = json.loads(body) if body else {}
        except Exception:
            req = {}
        src = (req.get("source") or "").strip()
        dst = (req.get("destination") or "").strip()
        if not src or not dst:
            self._send(400, "application/json",
                       json.dumps({"error": "source and destination required"}).encode())
            return
        a, p = resolve_model(src, installed_models())
        if not p:
            self._send(404, "application/json",
                       json.dumps({"error": "source '%s' not found" % src}).encode())
            return
        import shutil
        try:
            shutil.copy(p, os.path.join(MODEL_DIR, dst + ".gguf"))
        except OSError as e:
            self._send(500, "application/json", json.dumps({"error": str(e)}).encode())
            return
        self._send(200, "text/plain", b"copied\n")

    def _ollama_create(self, body):
        # vgpu has no Modelfile; create aliases an existing GGUF (from 'files' or 'from').
        try:
            req = json.loads(body) if body else {}
        except Exception:
            req = {}
        name = (req.get("model") or "").strip()
        if not name:
            self._send(400, "application/json",
                       json.dumps({"error": "no 'model' name specified"}).encode())
            return
        dst = os.path.join(MODEL_DIR, name + ".gguf")
        if os.path.exists(dst):
            self._send(200, "application/json", json.dumps({"status": "success"}).encode())
            return
        src = ""
        if req.get("from"):
            _, src = resolve_model(req["from"], installed_models())
        elif req.get("files"):
            # accept the first provided GGUF file name as a path under MODEL_DIR
            for fn in req["files"].values():
                if isinstance(fn, str) and fn.endswith(".gguf"):
                    cand = fn if os.path.isabs(fn) else os.path.join(MODEL_DIR, os.path.basename(fn))
                    if os.path.exists(cand):
                        src = cand
                        break
        if not src:
            self._send(404, "application/json",
                       json.dumps({"error": "no source GGUF found for create"}).encode())
            return
        import shutil
        try:
            shutil.copy(src, dst)
        except OSError as e:
            self._send(500, "application/json", json.dumps({"error": str(e)}).encode())
            return
        self._send(200, "application/json", json.dumps({"status": "success"}).encode())

    def _ollama_pull(self, body):
        # Delegate to `vgpu pull <model>` (supports alias / org/repo / url).
        try:
            req = json.loads(body) if body else {}
        except Exception:
            req = {}
        name = req.get("model")
        if not name:
            self._send(400, "application/json",
                       json.dumps({"error": "no 'model' specified"}).encode())
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        try:
            proc = subprocess.Popen(["%(d)s/vgpu" % {"d": os.path.dirname(os.path.abspath(__file__))},
                                     "pull", name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for raw in proc.stdout:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if line:
                    self.wfile.write((json.dumps({"status": line}) + "\n").encode())
                    self.wfile.flush()
            proc.wait()
        except Exception as e:
            self.wfile.write((json.dumps({"status": "error: %s" % e}) + "\n").encode())
        self.wfile.write((json.dumps({"status": "success"}) + "\n").encode())
        try:
            self.wfile.flush()
        except Exception:
            pass

    def _models(self):
        gw = self.server.gw
        models = installed_models()
        with gw.lock:
            cur = gw.current
            mode = gw.mode
        data = []
        if cur and cur in models:
            data.append({"id": cur, "object": "model", "owned_by": "vgpu",
                         "root": cur, "path": models[cur], "loaded": True, "mode": mode})
        for a in sorted(models):
            if a == cur:
                continue
            data.append({"id": a, "object": "model", "owned_by": "vgpu",
                         "root": a, "path": models[a], "loaded": False})
        out = {"object": "list", "data": data}
        self._send(200, "application/json", json.dumps(out).encode())

    # -- Ollama-native /api/* support --------------------------------------
    def _ollama_infer(self, openai_path, req_body):
        """Run an OpenAI-style inference and reshape the response to Ollama NDJSON.

        The backend emits OpenAI SSE; Ollama expects newline-delimited JSON
        objects ({...}\\n{...}) with a final object carrying done_reason.
        We also translate request/response field names.
        """
        gw = self.server.gw
        stream = bool(req_body.get("stream", True))
        # Ensure a model is resident (warm-load or swap) before proxying.
        with gw.lock:
            mode, err = gw.ensure_model(req_body.get("model") or gw.default_model)
            if err:
                self._send(503, "application/json", json.dumps({"error": err}).encode())
                return
            ka = req_body.get("keep_alive")
            ka_secs = gw.parse_keep_alive(ka) if ka is not None else gw._server_default_keep_alive
            ka_immediate = (ka_secs is not None and ka_secs <= 0)
        # Warm-load: an empty prompt/messages just loads the model (Ollama
        # returns done_reason "load" with no generation). Used by clients that
        # preload a model on selection.
        msgs = req_body.get("messages") if openai_path == "/v1/chat/completions" \
            else req_body.get("prompt", "")
        if (not msgs) or (isinstance(msgs, (list, str)) and len(msgs) == 0):
            gw._schedule_unload(ka_secs)
            created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                self.wfile.write((json.dumps({"model": gw.current,
                                               "created_at": created, "done": True,
                                               "done_reason": "load"}) + "\n").encode())
                self.wfile.flush()
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                if openai_path == "/v1/chat/completions":
                    self.wfile.write((json.dumps({"model": gw.current, "created_at": created,
                                                   "message": {"role": "assistant", "content": ""},
                                                   "done": True, "done_reason": "load"}).encode()))
                else:
                    self.wfile.write((json.dumps({"model": gw.current, "created_at": created,
                                                   "response": "", "done": True,
                                                   "done_reason": "load"}).encode()))
                self.wfile.flush()
            return
        # Forward to backend as OpenAI request.
        fwd = {"model": gw.current, "stream": True}
        if openai_path == "/v1/chat/completions":
            fwd["messages"] = req_body.get("messages", [])
        else:
            fwd["prompt"] = req_body.get("prompt", "")
        # Pass through generation params translated from Ollama 'options'.
        for k in ("max_tokens", "temperature", "top_p", "top_k", "seed", "stop"):
            if k in req_body and req_body[k] is not None:
                fwd[k] = req_body[k]
        if "response_format" in req_body:
            fwd["response_format"] = req_body["response_format"]
        body = json.dumps(fwd).encode()
        try:
            c = HTTPConnection(gw.backend_host, gw.backend_port, timeout=600)
            hdrs = {"Content-Type": "application/json", "Content-Length": str(len(body))}
            c.request("POST", openai_path, body=body, headers=hdrs)
            r = c.getresponse()
        except Exception as e:
            self._send(502, "application/json",
                       json.dumps({"error": "backend proxy error: %s" % e}).encode())
            return

        if r.status != 200:
            payload = r.read()
            try:
                self._send(r.status, "application/json", payload)
            except Exception:
                pass
            return

        model = gw.current
        created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        is_chat = openai_path == "/v1/chat/completions"

        def emit(obj):
            try:
                self.wfile.write((json.dumps(obj) + "\n").encode())
                self.wfile.flush()
            except Exception:
                pass

        if not stream:
            # Buffer full OpenAI reply, emit a single Ollama object.
            full = ""
            done_reason = "stop"
            while True:
                line = r.readline()
                if not line:
                    break
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                try:
                    o = json.loads(data)
                except Exception:
                    continue
                try:
                    fr = o.get("choices", [{}])[0].get("finish_reason")
                    if fr:
                        done_reason = fr
                except Exception:
                    pass
                if is_chat:
                    full += (o.get("choices", [{}])[0].get("delta", {}).get("content") or "")
                else:
                    full += (o.get("choices", [{}])[0].get("text") or "")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if is_chat:
                emit({"model": model, "created_at": created,
                      "message": {"role": "assistant", "content": full},
                      "done": True, "done_reason": done_reason})
            else:
                emit({"model": model, "created_at": created,
                      "response": full, "done": True, "done_reason": done_reason})
            gw._schedule_unload(0 if ka_immediate else ka_secs)
            c.close()
            return

        # Streaming: forward reshaped NDJSON objects.
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        try:
            while True:
                line = r.readline()
                if not line:
                    break
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                try:
                    o = json.loads(data)
                except Exception:
                    continue
                ch = o.get("choices", [{}])[0]
                if is_chat:
                    content = ch.get("delta", {}).get("content") or ""
                    emit({"model": model, "created_at": created,
                          "message": {"role": "assistant", "content": content},
                          "done": False})
                else:
                    text = ch.get("text") or ""
                    emit({"model": model, "created_at": created,
                          "response": text, "done": False})
        except Exception:
            pass
        if is_chat:
            emit({"model": model, "created_at": created,
                  "message": {"role": "assistant", "content": ""}, "done": True,
                  "done_reason": "stop"})
        else:
            emit({"model": model, "created_at": created,
                  "response": "", "done": True, "done_reason": "stop"})
        gw._schedule_unload(0 if ka_immediate else ka_secs)
        c.close()


class GatewayServer(ThreadingHTTPServer):
    def __init__(self, addr, gw):
        super().__init__(addr, Handler)
        self.gw = gw
        self.daemon_threads = True
        self.allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public-host", default="0.0.0.0")
    ap.add_argument("--public-port", type=int, default=int(os.environ.get("PORT", 11434)))
    ap.add_argument("--backend-host", default="127.0.0.1")
    ap.add_argument("--backend-port", type=int, default=11435)
    ap.add_argument("--vram", default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--min-free", default=None,
                    help="keep this many GB of RAM free after launch; refuse to start "
                         "a backend that wouldn't fit in MemAvailable (+ margin) to avoid an OOM kill")
    ap.add_argument("--no-oom-gate", action="store_true",
                    help="disable the OOM gate and launch regardless of free RAM (old behavior)")
    ap.add_argument("--default-model", default=None)
    ap.add_argument("--keep-alive", default=None,
                    help="default idle lifetime for a loaded model (Ollama semantics: "
                         "0=unload now, -1=forever, 5m/30s/seconds=idle timeout; default 5m)")
    args = ap.parse_args()

    gw = Gateway(args.backend_host, args.backend_port,
                 args.public_host, args.public_port,
                 args.vram, args.cpu, args.default_model, args.keep_alive,
                 min_free=args.min_free, no_oom_gate=args.no_oom_gate)

    def cleanup(*a):
        gw.stop_backend()
        os._exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)   # daemon: survive parent shell exit
    except (AttributeError, ValueError):
        pass

    srv = GatewayServer((args.public_host, args.public_port), gw)
    sys.stderr.write("vgpu gateway listening on %s:%s (backend %s:%s)\n" % (
        args.public_host, args.public_port, args.backend_host, args.backend_port))
    sys.stderr.flush()

    # Warm up the default model in the background so the first request is instant.
    if args.default_model:
        def warm():
            with gw.lock:
                gw.ensure_model(args.default_model)
        threading.Thread(target=warm, daemon=True).start()

    try:
        srv.serve_forever()
    finally:
        gw.stop_backend()


if __name__ == "__main__":
    main()
