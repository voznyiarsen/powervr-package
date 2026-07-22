#!/usr/bin/env python3
"""vgpu - ollama-style CLI for direct llama-server Vulkan inference (non-K-quant GGUFs).

Subcommands: pull, add, search, list, rm, serve, run, gpu, ps, stop, help
GPU path mirrors vulkan-serve.sh (verified working on PowerVR DXT-48-1536 MC1).

Color is uv-style: controlled by --color auto|always|never, --no-color, or the
VG_COLOR / NO_COLOR / FORCE_COLOR / CLICOLOR_FORCE environment variables.

This is the Python port of the original bash CLI; all shared logic lives in
vgpu_core.py (which the gateway also imports from).
"""

import os
import sys
import json
import glob
import shutil
import argparse
import subprocess
import urllib.request

import vgpu_core as core

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Global color flags (parsed before the subcommand, uv-style)
# ---------------------------------------------------------------------------
def parse_global_color(argv):
    """Pull --color/--no-color off the front; return (vg_color, rest)."""
    vg_color = os.environ.get("VG_COLOR", "auto")
    rest = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--color":
            if i + 1 < len(argv):
                vg_color = argv[i + 1]
                i += 2
            else:
                vg_color = "auto"
                i += 1
        elif a.startswith("--color="):
            vg_color = a.split("=", 1)[1]
            i += 1
        elif a == "--no-color":
            vg_color = "never"
            i += 1
        else:
            rest.append(a)
            i += 1
    return vg_color, rest


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------
def cmd_pull(args, c):
    core.init_registry()
    if not args.targets:
        sys.stderr.write(c.bold("usage:") + " vgpu pull "
                         + c.cyan("<alias|org/repo/file.gguf|url>") + " [more...]\n")
        return 1
    if shutil_which("aria2c") is None:
        c.error("aria2c is required for downloads (install the aria2 package)")
        return 1
    rc = 0
    for arg in args.targets:
        alias, url = core.resolve_url(arg)
        if not url:
            c.error("unknown model alias: %s" % c.cyan(arg))
            sys.stderr.write(c.dim("Known aliases:") + "\n")
            reg = core.read_registry()
            for a in reg:
                sys.stderr.write("  " + c.cyan(a) + "\n")
            rc = 1
            continue
        dest = core.model_path(alias)
        if os.path.isfile(dest):
            sys.stderr.write(c.cyan(alias) + c.dim(" already downloaded (%s)\n" % dest))
            continue
        threads = os.cpu_count() or 4
        sys.stderr.write(c.bold("Downloading") + " " + c.cyan(alias) + " "
                         + c.dim("-> %s (%d threads)" % (dest, threads)) + "\n")
        auth = []
        if os.environ.get("HF_TOKEN"):
            auth = ["--header=Authorization: Bearer %s" % os.environ["HF_TOKEN"]]
        ddir = os.path.dirname(dest)
        dfile = os.path.basename(dest)
        # Progress reporting: let aria2c inherit the terminal so it can draw a
        # live progress bar (it only does this when stdout/stderr is a TTY).
        # --summary-interval=1 forces periodic progress lines; --console-log-level=warn
        # keeps the noise down to progress + errors. When stdout is captured
        # (piped/headless) aria2c falls back to its Download Results summary.
        # Note: the Termux aria2c build has no -# flag, so we rely on the
        # default TTY bar instead.
        r = subprocess.run(["aria2c", "-x", str(threads), "-s", str(threads),
                            "-c", "--file-allocation=none",
                            "--summary-interval=1", "--console-log-level=warn"] + auth
                           + ["-d", ddir, "-o", dfile, url])
        if r.returncode == 0:
            sys.stderr.write(c.green("Done") + ": " + c.cyan(alias) + "\n")
        else:
            c.error("download failed: %s" % c.cyan(alias))
            for p in (dest, dest + ".aria2"):
                try:
                    os.remove(p)
                except OSError:
                    pass
            rc = 1
    return rc


def cmd_add(args, c):
    if not args.alias or not args.url:
        sys.stderr.write(c.bold("usage:") + " vgpu add "
                         + c.cyan("<alias>") + " " + c.cyan("<url>") + "\n")
        return 1
    core.add_alias(args.alias, args.url)
    sys.stderr.write(c.green("Registered") + " " + c.cyan(args.alias) + "\n")
    return 0


def cmd_search(args, c):
    query = args.query
    limit = args.limit
    sys.stderr.write(c.dim("Searching HuggingFace for ") + "'" + c.cyan(query)
                     + "' " + c.dim("(limit %d)..." % limit) + "\n")
    try:
        import urllib.parse
        url = ("https://huggingface.co/api/models?full=true&"
               + urllib.parse.urlencode({"search": query, "limit": limit}))
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "vgpu"})
        data = urllib.request.urlopen(req, timeout=30).read()
    except Exception as e:
        c.error("search failed (network error?): %s" % e)
        return 1
    try:
        items = json.loads(data)
    except Exception:
        c.error("no response from HuggingFace.")
        return 1
    CY = core.C(True).CYAN if c.use else ""
    GR = core.C(True).GREEN if c.use else ""
    DI = core.C(True).DIM if c.use else ""
    RS = core.C(True).RESET if c.use else ""
    found = 0
    for m in items:
        sid = m.get("id", "")
        sibs = m.get("siblings") or []
        ggufs = [s.get("rfilename") for s in sibs
                 if (s.get("rfilename") or "").lower().endswith(".gguf")]
        if not ggufs:
            continue
        found += 1
        print(CY + sid + RS)
        for i, g in enumerate(ggufs):
            if i >= 15:
                print(DI + "  ... +%d more GGUF file(s)" % (len(ggufs) - 15) + RS)
                break
            print(DI + "  %s" % g + RS)
            print(GR + "      vgpu pull %s/%s" % (sid, g) + RS)
    if found == 0:
        print(DI + "No GGUF-containing models found for that query." + RS)
    return 0


def cmd_list(args, c):
    core.init_registry()
    sys.stderr.write(c.bold("Downloaded models") + " " + c.dim("in")
                     + " " + c.cyan(core.MODELDIR) + ":\n")
    found = 0
    for f in sorted(glob.glob(os.path.join(core.MODELDIR, "*.gguf"))):
        found = 1
        name = ("%-28s" % os.path.basename(f)[:-5])
        try:
            size = subprocess.check_output(["du", "-h", f]).decode().split()[0]
        except Exception:
            size = "?"
        sys.stderr.write("  " + c.cyan(name) + " " + c.dim(size) + "\n")
    if not found:
        sys.stderr.write("  " + c.dim("(none)") + "\n")
    sys.stderr.write(c.bold("Known aliases") + " " + c.dim("(not downloaded):") + "\n")
    for a in core.read_registry():
        sys.stderr.write("  " + c.cyan(a) + "\n")
    return 0


def cmd_rm(args, c):
    if not args.aliases:
        sys.stderr.write(c.bold("usage:") + " vgpu rm " + c.cyan("<alias>")
                         + " [more...]\n")
        return 1
    rc = 0
    for a in args.aliases:
        f = core.model_path(a)
        if os.path.isfile(f):
            os.remove(f)
            sys.stderr.write(c.green("Removed") + " " + c.cyan(a) + "\n")
        else:
            c.error("%s not found" % c.cyan(a))
            rc = 1
    return rc


def cmd_gpu(args, c):
    sys.stderr.write(c.bold("Vulkan devices reachable from this process:") + "\n")
    out, rc = core.gpu_probe()
    if not out:
        sys.stderr.write("  " + c.red("none") + c.dim(" (probe failed to build or run)\n"))
        rc = 1
    else:
        for line in out.splitlines():
            sys.stderr.write("  " + line + "\n")
    if rc == 0:
        sys.stderr.write(c.green("OK: a hardware Vulkan GPU is reachable.") + "\n")
    else:
        sys.stderr.write(c.red("NO hardware Vulkan GPU reachable.") + " "
                         + c.dim("Models will run on CPU.") + "\n")
        sys.stderr.write(c.dim("Fix: install the Android Vulkan loader (it bridges to /vendor .../vulkan.<soc>.so):") + "\n")
        sys.stderr.write("  " + c.cyan("pkg install vulkan-loader-android")
                         + c.dim("  # or: dpkg -i vulkan-loader-android_*.deb") + "\n")
        sys.stderr.write(c.dim("and ensure OLLAMA_VULKAN is not 0 (vgpu forces it on automatically).") + "\n")
    return rc


def cmd_serve(args, c):
    if core.server_alive():
        sys.stderr.write(c.yellow("A server is already running") + " on port "
                         + c.cyan(str(core.read_port())) + ". Stop it with: "
                         + c.green("vgpu stop") + "\n")
        return 1
    if not args.cpu and not core.gpu_reachable():
        c.warn("No hardware Vulkan GPU reachable right now (see: "
               + c.cyan("vgpu gpu") + "). The backend will try the GPU and "
               "auto-fall back to CPU if it fails.")

    # --- Fixed single-model server ---
    if args.alias:
        if not core.model_installed(args.alias, c):
            return 1
        f = core.model_path(args.alias)
        ctx = core.gate_ctx(f, core.DEFAULT_CTX, args.vram, c)
        if ctx is None:
            return 1
        if not core.gate_oom(f, ctx, args.min_free, args.no_oom_gate, c):
            return 1
        mode = core.start_with_fallback(f, args.port, core.SERVE_LOG, args.cpu,
                                        args.alias, c, ctx)
        if mode is None:
            return 1
        core.server_model_set(args.alias)
        if mode == "gpu":
            sys.stderr.write(c.green("Ready") + " " + c.dim("(GPU, single model %s). OpenAI API at:" % args.alias)
                             + " " + c.cyan("http://%s:%d/v1/chat/completions" % (args.host, args.port)) + "\n")
        else:
            sys.stderr.write(c.green("Ready") + " " + c.yellow("(CPU fallback, single model %s)." % args.alias)
                             + " " + c.dim("OpenAI API at:") + " "
                             + c.cyan("http://%s:%d/v1/chat/completions" % (args.host, args.port)) + "\n")
        return 0

    # --- Multi-model server (Ollama-like: load/unload on the fly) ---
    gw_py = os.path.join(SCRIPT_DIR, "vgpu_gateway.py")
    cmd = [sys.executable, "-u", gw_py, "--public-host", args.host,
           "--public-port", str(args.port), "--backend-port", str(args.backend_port)]
    if args.vram:
        cmd += ["--vram", args.vram]
    if args.cpu:
        cmd += ["--cpu"]
    if args.min_free:
        cmd += ["--min-free", str(args.min_free)]
    if args.no_oom_gate:
        cmd += ["--no-oom-gate"]
    if args.keep_alive:
        cmd += ["--keep-alive", args.keep_alive]
    if args.default:
        cmd += ["--default-model", args.default]
    proc = subprocess.Popen(cmd, stdout=open(core.GATEWAY_LOG, "a"),
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
    pid = proc.pid
    with open(core.PIDFILE, "w") as f:
        f.write(str(pid))
    with open(core.PORTFILE, "w") as f:
        f.write(str(args.port))
    core.server_model_set("gateway")
    import time
    for _ in range(30):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/healthz" % args.port,
                                   timeout=2)
            if args.default:
                sys.stderr.write(c.green("Ready") + " " + c.dim("(multi-model; warmed")
                                 + " " + c.cyan(args.default) + c.dim("). OpenAI API at:")
                                 + " " + c.cyan("http://%s:%d/v1/chat/completions" % (args.host, args.port)) + "\n")
            else:
                sys.stderr.write(c.green("Ready") + " " + c.dim("(multi-model; name a model per request). OpenAI API at:")
                                 + " " + c.cyan("http://%s:%d/v1/chat/completions" % (args.host, args.port)) + "\n")
            return 0
        except Exception:
            pass
        if pid and not _pid_alive(pid):
            c.error("server exited early:")
            _tail(core.GATEWAY_LOG, 20)
            try:
                os.remove(core.PIDFILE)
            except OSError:
                pass
            return 1
        time.sleep(0.5)
    c.error("server did not become ready:")
    _tail(core.GATEWAY_LOG, 20)
    return 1


def _cmd_run_headless(args, c, f, port):
    """Non-interactive path for `vgpu run <alias> "PROMPT"`.

    If a server is already running we attach to it (like the interactive path)
    and leave it intact -- just send one prompt and exit. Otherwise we start a
    backend, wait for readiness, send the prompt, print the reply, and tear the
    backend down so the command is truly one-shot.
    """
    if core.server_alive():
        rport = core.read_port()
        if core.server_is_gateway(rport):
            sys.stderr.write(c.green("Attaching") + " to running "
                             + c.cyan("vgpu serve") + c.dim(" (multi-model) on port")
                             + " " + c.cyan(str(rport)) + " ...\n")
            rc = _run_prompt(rport, args.alias, args.prompt, args.no_stream, c)
            sys.stderr.write(c.dim("Left the running server intact.") + "\n")
            return rc
        running = core.server_model_get()
        if running and running != args.alias and running != "gateway":
            c.warn("A single-model server for %s is already running; it will "
                   "answer as %s, not %s." % (c.cyan(running), c.cyan(running),
                                              c.cyan(args.alias)))
        else:
            sys.stderr.write(c.green("Attaching") + " to running server on port "
                             + c.cyan(str(rport)) + " ...\n")
        rc = _run_prompt(rport, args.alias, args.prompt, args.no_stream, c)
        sys.stderr.write(c.dim("Left the running server intact.") + "\n")
        return rc

    if not args.cpu and not core.gpu_reachable():
        c.warn("No hardware Vulkan GPU is reachable right now (see: "
               + c.cyan("vgpu gpu") + "). The backend will try the GPU and "
               "auto-fall back to CPU if it fails.")
    ctx = core.gate_ctx(f, core.DEFAULT_CTX, args.vram, c)
    if ctx is None:
        return 1
    if not core.gate_oom(f, ctx, args.min_free, args.no_oom_gate, c):
        return 1
    mode = core.start_with_fallback(f, port, core.SERVE_LOG, args.cpu,
                                    args.alias, c, ctx)
    if mode is None:
        return 1
    core.server_model_set(args.alias)
    try:
        rc = _run_prompt(port, args.alias, args.prompt, args.no_stream, c)
    finally:
        if core.server_alive():
            try:
                with open(core.PIDFILE) as pf:
                    os.kill(int(pf.read().strip()), 15)
            except Exception:
                pass
            import time
            time.sleep(1)
        core.server_model_clear()
        try:
            os.remove(core.PIDFILE)
        except OSError:
            pass
        sys.stderr.write(c.dim("Stopping server.") + "\n")
    return rc


def cmd_run(args, c):
    # Disambiguate the legacy 2nd positional: an int is a port, anything else
    # is a one-shot prompt (headless mode). Explicit --port always wins.
    if args.prompt is not None and args.port is None and args.prompt.isdigit():
        args.port = int(args.prompt)
        args.prompt = None

    if not args.alias:
        if not sys.stdin.isatty():
            sys.stderr.write(c.bold("usage:") + " vgpu run [--cpu] [--vram <GB>] "
                             + "[--no-stream] " + c.cyan("<alias>") + " [prompt]\n")
            return 1
        core.init_registry()
        selected = select_model(c)
        if not selected:
            return 1
        args.alias = selected

    if not core.model_installed(args.alias, c):
        return 1
    f = core.model_path(args.alias)
    port = args.port if args.port else core.DEFAULT_PORT

    # --- Headless one-shot: vgpu run <alias> "PROMPT" ---
    if args.prompt is not None:
        return _cmd_run_headless(args, c, f, port)

    # --- Attach to a running server (vgpu serve), like `ollama run` ---
    if core.server_alive():
        rport = core.read_port()
        chat = os.path.join(SCRIPT_DIR, "vgpu_chat.py")
        if core.server_is_gateway(rport):
            sys.stderr.write(c.green("Attaching") + " to running "
                             + c.cyan("vgpu serve") + c.dim(" (multi-model) on port")
                             + " " + c.cyan(str(rport)) + " ...\n")
            _run_chat(chat, rport, args.alias, args.no_stream)
            sys.stderr.write(c.dim("Left the running server intact (it was started "
                                   "by 'vgpu serve').") + "\n")
            return 0
        running = core.server_model_get()
        if running and running != args.alias and running != "gateway":
            c.warn("A single-model server for %s is already running on port %s; "
                   "it cannot load %s." % (c.cyan(running), c.cyan(str(rport)),
                                           c.cyan(args.alias)))
            sys.stderr.write(c.dim("Attaching anyway -- you'll chat with %s. "
                                   "To switch: vgpu stop && vgpu run %s" %
                                   (running, args.alias)) + "\n")
        else:
            sys.stderr.write(c.green("Attaching") + " to running server on port "
                             + c.cyan(str(rport)) + " ...\n")
        _run_chat(chat, rport, args.alias, args.no_stream)
        sys.stderr.write(c.dim("Left the running server intact.") + "\n")
        return 0

    if not args.cpu and not core.gpu_reachable():
        c.warn("No hardware Vulkan GPU is reachable right now (see: "
               + c.cyan("vgpu gpu") + "). The backend will try the GPU and "
               "auto-fall back to CPU if it fails.")
    ctx = core.gate_ctx(f, core.DEFAULT_CTX, args.vram, c)
    if ctx is None:
        return 1
    if not core.gate_oom(f, ctx, args.min_free, args.no_oom_gate, c):
        return 1
    mode = core.start_with_fallback(f, port, core.SERVE_LOG, args.cpu,
                                    args.alias, c, ctx)
    if mode is None:
        return 1
    core.server_model_set(args.alias)
    import signal
    def _on_int(signum, frame):
        if core.server_alive():
            try:
                with open(core.PIDFILE) as pf:
                    os.kill(int(pf.read().strip()), 15)
            except Exception:
                pass
        core.server_model_clear()
        try:
            os.remove(core.PIDFILE)
        except OSError:
            pass
        sys.exit(0)
    signal.signal(signal.SIGINT, _on_int)
    if mode == "cpu":
        sys.stderr.write(c.yellow(">>> Running on CPU (NOT the GPU). Expect slow generation. <<<") + "\n")
    sys.stderr.write(c.green("Ready") + ". " + c.dim("Chat below (type /exit to quit).") + "\n")
    chat = os.path.join(SCRIPT_DIR, "vgpu_chat.py")
    _run_chat(chat, args.port, args.alias, args.no_stream)
    if core.server_alive():
        try:
            with open(core.PIDFILE) as pf:
                os.kill(int(pf.read().strip()), 15)
        except Exception:
            pass
        import time
        time.sleep(1)
    core.server_model_clear()
    try:
        os.remove(core.PIDFILE)
    except OSError:
        pass
    sys.stderr.write(c.dim("Stopping server.") + "\n")
    return 0


def _backend_label(backend, c):
    """Human-readable, colorized backend tag for `vgpu ps`."""
    if backend == "gpu":
        return c.green("backend: GPU (Vulkan)")
    if backend == "cpu":
        return c.yellow("backend: CPU (fallback)")
    return c.dim("backend: unknown")


def cmd_ps(args, c):
    if core.server_alive():
        port = core.read_port()
        backend = core.server_backend()
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/vgpu/status" % port,
                                        timeout=2) as r:
                st = json.loads(r.read())
        except Exception:
            st = None
        pid = _read_pidfile()
        if st:
            cur = st.get("current_model") or "(none)"
            mode = st.get("mode") or ""
            s = (c.green("vgpu gateway running") + ": pid " + c.cyan(str(pid))
                 + ", port " + c.cyan(str(port)) + ", model " + c.cyan(str(cur)))
            if mode:
                s += " (%s)" % mode
            s += "\n  " + _backend_label(backend, c)
            sys.stderr.write(s + "\n")
        else:
            s = (c.green("vgpu server running") + ": pid " + c.cyan(str(pid))
                 + ", port " + c.cyan(str(port)))
            s += "\n  " + _backend_label(backend, c)
            sys.stderr.write(s + "\n")
    else:
        sys.stderr.write(c.dim("no server running") + "\n")
    return 0


def cmd_stop(args, c):
    core.stop_server(c)
    return 0


def cmd_help(args, c):
    print(HELP_TEXT.format(c=c))
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run_chat(chat, port, model, no_stream):
    cmd = [sys.executable, "-u", chat, str(port), "--model", model]
    if no_stream:
        cmd.append("--no-stream")
    subprocess.run(cmd)


def _run_prompt(port, model, prompt, no_stream, c):
    """Send a single prompt to a live server and print the reply (headless).

    Used by `vgpu run <alias> "PROMPT"` so the user gets one answer and the
    process exits without ever starting the interactive REPL. Mirrors the
    request shape vgpu_chat.py uses against /v1/chat/completions.
    """
    url = "http://127.0.0.1:%d/v1/chat/completions" % port
    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": not no_stream,
    }).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=600)
    except Exception as e:
        c.error("request failed: %s" % e)
        return 1
    if no_stream:
        try:
            obj = json.load(resp)
            text = obj["choices"][0]["message"]["content"]
        except Exception as e:
            c.error("bad response: %s" % e)
            return 1
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
        return 0
    # Streaming: print token deltas as they arrive (no "[DONE]" noise).
    full = []
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        try:
            delta = obj["choices"][0]["delta"].get("content", "")
        except Exception:
            continue
        if delta:
            sys.stdout.write(delta)
            sys.stdout.flush()
            full.append(delta)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_pidfile():
    try:
        with open(core.PIDFILE) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _find_gateway_pid(gw_py):
    # Best-effort: the gateway is launched detached; we recorded nothing yet.
    # Fall back to scanning processes for vgpu_gateway.py (Termux-friendly).
    try:
        for p in os.listdir("/proc"):
            if not p.isdigit():
                continue
            try:
                with open("/proc/%s/cmdline" % p, "rb") as f:
                    cl = f.read().decode("utf-8", "replace")
            except Exception:
                continue
            if "vgpu_gateway.py" in cl and str(os.getpid()) != p:
                return int(p)
    except Exception:
        pass
    return None


def _tail(path, n=15):
    try:
        with open(path) as f:
            lines = f.read().splitlines()[-n:]
        for ln in lines:
            sys.stderr.write(ln + "\n")
    except Exception:
        pass


def select_model(c):
    models = sorted(installed_models().keys())
    if not models:
        sys.stderr.write(c.yellow("No models installed yet.") + " "
                         + c.dim("Download one with:") + " "
                         + c.green("vgpu pull <alias>") + "\n")
        sys.stderr.write(c.dim("Known aliases:") + "\n")
        for a in core.read_registry():
            sys.stderr.write("  " + c.cyan(a) + "\n")
        return None
    sys.stderr.write(c.bold("Select a model to run:") + "\n")
    for i, m in enumerate(models, 1):
        sys.stderr.write("  " + c.cyan(str(i)) + ") " + c.green(m) + "\n")
    sys.stderr.write(c.bold("→ ") )
    try:
        choice = sys.stdin.readline()
    except Exception:
        sys.stderr.write(c.dim("cancelled") + "\n")
        return None
    if choice is None:
        sys.stderr.write(c.dim("cancelled") + "\n")
        return None
    choice = choice.strip().replace(" ", "")
    if not choice:
        sys.stderr.write(c.dim("cancelled") + "\n")
        return None
    if choice.isdigit() and 1 <= int(choice) <= len(models):
        return models[int(choice) - 1]
    if choice in models:
        return choice
    c.error("invalid selection: %s" % c.cyan(choice))
    return None


def shutil_which(name):
    return shutil.which(name)


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------
HELP_TEXT = """\
{c.GREEN}vgpu{c.RESET} - Vulkan GPU inference (ollama-like CLI)

  {c.GREEN}vgpu{c.RESET} {c.CYAN}{c.BOLD}pull{c.RESET}   {c.CYAN}<alias|org/repo/file.gguf|url>{c.RESET} [more...]   download one or more GGUFs (non-K-quant only)
  {c.GREEN}vgpu{c.RESET} {c.CYAN}{c.BOLD}add{c.RESET}    {c.CYAN}<alias> <url>{c.RESET}                     register a custom model alias
  {c.GREEN}vgpu{c.RESET} {c.CYAN}{c.BOLD}search{c.RESET} {c.CYAN}<query>{c.RESET} [limit]                 search HuggingFace for GGUF models
  {c.GREEN}vgpu{c.RESET} {c.CYAN}{c.BOLD}list{c.RESET}                                   show downloaded + known models
  {c.GREEN}vgpu{c.RESET} {c.CYAN}{c.BOLD}rm{c.RESET}     {c.CYAN}<alias>{c.RESET} [more...]                 delete one or more downloaded GGUFs
  {c.GREEN}vgpu{c.RESET} {c.CYAN}{c.BOLD}serve{c.RESET}  [--cpu] [--vram <GB>] [--min-free <GB>] [--no-oom-gate] [--keep-alive <dur>] [--host/--port/--backend-port] [{c.CYAN}<alias>{c.RESET} [port]]   Ollama-like server: runs one model, or (no alias) loads/unloads models on demand
  {c.GREEN}vgpu{c.RESET} {c.CYAN}{c.BOLD}run{c.RESET}    [--cpu] [--vram <GB>] [--min-free <GB>] [--no-oom-gate] [--no-stream] {c.CYAN}<alias>{c.RESET} [{c.CYAN}<port>|"PROMPT"{c.RESET}]   interactive chat (streams); {c.CYAN}vgpu run <alias> "PROMPT"{c.RESET} runs headless one-shot; menu if no alias; attaches to a running 'vgpu serve'
  {c.GREEN}vgpu{c.RESET} {c.CYAN}{c.BOLD}gpu{c.RESET}                                            list reachable Vulkan devices / confirm GPU is visible
  {c.GREEN}vgpu{c.RESET} {c.CYAN}{c.BOLD}ps{c.RESET}                                     show running server
  {c.GREEN}vgpu{c.RESET} {c.CYAN}{c.BOLD}stop{c.RESET}                                   stop running server
  {c.GREEN}vgpu{c.RESET} {c.CYAN}{c.BOLD}help{c.RESET}                                   this message

GPU works only with non-K-quant GGUFs (Q4_0 / Q5_0 / Q8_0 / F16 / IQ4_XS).
If the Vulkan backend fails to start (e.g. K-quants), it auto-falls back to CPU with a warning.
Silently-corrupting quants (e.g. IQ3_XS) do NOT crash -- run those with: {c.GREEN}vgpu run --cpu <alias>{c.RESET}
{c.GREEN}vgpu run{c.RESET} streams tokens (OpenAI SSE) by default; use --no-stream to wait for the full reply. {c.GREEN}vgpu run <alias> "PROMPT"{c.RESET} runs one prompt headlessly (no REPL) and exits; if a server is already running it attaches and leaves it intact, otherwise it starts a backend, answers, and stops it.
If {c.GREEN}vgpu serve{c.RESET} is already running, {c.GREEN}vgpu run <alias>{c.RESET} attaches to it (like {c.GREEN}ollama run{c.RESET}) and does NOT stop it on exit; for the multi-model server the requested model is loaded/swapped on the fly.
{c.GREEN}--vram <GB>{c.RESET} optionally caps the model's RAM footprint (weights + KV cache); context auto-shrinks to fit, or it refuses if the model can't.
{c.GREEN}--min-free <GB>{c.RESET} (default 0) keeps this much RAM free after launch; the OOM gate refuses to start a backend that wouldn't fit in MemAvailable (+ this margin), to avoid an OOM kill. Set {c.GREEN}--no-oom-gate{c.RESET} to launch regardless (old behavior).
{c.GREEN}vgpu serve{c.RESET} (no alias) is Ollama-like: it keeps ONE backend and auto-loads the model named in each request, unloading the previous one. At most one model's weights live in RAM at a time (safe under a VRAM budget). Control idle lifetime with {c.GREEN}--keep-alive <dur>{c.RESET} (0=unload now, -1=forever, default 5m) or per-request {c.GREEN}keep_alive{c.RESET}. Use {c.GREEN}POST /v1/unload{c.RESET} (or {c.GREEN}/vgpu/unload{c.RESET}) to free the current model and {c.GREEN}GET /vgpu/status{c.RESET} for state.
Set HF_TOKEN to pull gated HuggingFace models.
"""


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------
def build_parser(c):
    p = argparse.ArgumentParser(prog="vgpu", add_help=False,
                                description="Vulkan GPU inference (ollama-like CLI)")
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("pull", help="download one or more GGUFs")
    sp.add_argument("targets", nargs="*")
    sp.set_defaults(func=cmd_pull)

    sp = sub.add_parser("add", help="register a custom model alias")
    sp.add_argument("alias")
    sp.add_argument("url")
    sp.set_defaults(func=cmd_add)

    sp = sub.add_parser("search", help="search HuggingFace for GGUF models")
    sp.add_argument("query")
    sp.add_argument("limit", nargs="?", default=20, type=int)
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("list", help="show downloaded + known models")
    sp.set_defaults(func=cmd_list)
    sp = sub.add_parser("ls", help="alias for 'list'")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("rm", help="delete one or more downloaded GGUFs")
    sp.add_argument("aliases", nargs="*")
    sp.set_defaults(func=cmd_rm)

    sp = sub.add_parser("gpu", help="list reachable Vulkan devices")
    sp.set_defaults(func=cmd_gpu)

    sp = sub.add_parser("serve", help="Ollama-like server")
    sp.add_argument("--cpu", action="store_true")
    sp.add_argument("--vram", default=None)
    sp.add_argument("--min-free", default=None)
    sp.add_argument("--no-oom-gate", action="store_true")
    sp.add_argument("--keep-alive", default=None)
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=core.DEFAULT_PORT)
    sp.add_argument("--backend-port", type=int, default=core.GATEWAY_BK_PORT)
    sp.add_argument("--default", default=None)
    sp.add_argument("alias", nargs="?")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("run", help="chat (interactive or one-shot); attaches to serve")
    sp.add_argument("--cpu", action="store_true")
    sp.add_argument("--vram", default=None)
    sp.add_argument("--min-free", default=None)
    sp.add_argument("--no-oom-gate", action="store_true")
    sp.add_argument("--no-stream", action="store_true")
    sp.add_argument("--port", type=int, default=None)
    sp.add_argument("alias", nargs="?")
    # Optional 2nd positional. If it parses as an int it is taken as the port
    # (legacy `vgpu run <alias> <port>` syntax); otherwise it is the one-shot
    # prompt for non-interactive, headless use: `vgpu run <alias> "PROMPT"`.
    sp.add_argument("prompt", nargs="?", default=None)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("ps", help="show running server")
    sp.set_defaults(func=cmd_ps)

    sp = sub.add_parser("stop", help="stop running server")
    sp.set_defaults(func=cmd_stop)

    sp = sub.add_parser("help", help="this message")
    sp.set_defaults(func=cmd_help)

    return p


def main(argv):
    vg_color, rest = parse_global_color(argv)
    use = core.resolve_color(vg_color)
    c = core.C(use)
    p = build_parser(c)
    args = p.parse_args(rest)
    if not getattr(args, "cmd", None):
        return cmd_help(args, c)
    if args.cmd in ("help",):
        return cmd_help(args, c)
    return args.func(args, c)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
