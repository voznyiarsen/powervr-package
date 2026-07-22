#!/usr/bin/env python3
# vgpu interactive chat REPL (kept as a file so stdin stays the user's pipe).
# Streams tokens by default (OpenAI SSE); pass --no-stream to wait for the full reply.
import sys, json, urllib.request, os

# uv-style palette. Inherit the parent vgpu script's decision via _VG_USE_COLOR,
# and respect NO_COLOR / FORCE_COLOR the same way the bash CLI does.
if os.environ.get("NO_COLOR"):
    _USE_COLOR = False
elif os.environ.get("FORCE_COLOR") or os.environ.get("_VG_USE_COLOR") == "1":
    _USE_COLOR = True
else:
    _USE_COLOR = sys.stdout.isatty()

def _c(code, s):
    return ("\033[%dm" % code + s + "\033[0m") if _USE_COLOR else s

cyan = lambda s: _c(36, s)
red  = lambda s: _c(31, s)
dim  = lambda s: _c(2,  s)

# Enable GNU readline line editing when attached to a real terminal so the user
# gets arrow-key cursor movement (left/right), Home/End, and up/down history
# recall. We only activate it for a TTY; for piped/redirected input (e.g. an
# automated `echo "/exit" | vgpu run`) we leave stdin alone so EOF behaves.
try:
    import readline as _rl
    _HAVE_RL = sys.stdin.isatty()
except Exception:
    _rl = None
    _HAVE_RL = False

def prompt():
    """The '>>> ' prompt, marked for readline when colors are on.

    GNU readline needs non-printing (color escape) sequences wrapped in
    \\001..\\002 so it counts prompt width correctly and the cursor stays
    aligned on long input lines."""
    p = cyan(">>> ")
    if _USE_COLOR and _HAVE_RL:
        return "\001" + p + "\002"
    return p

port = sys.argv[1]
no_stream = "--no-stream" in sys.argv[2:]
# Model to send with each request. For a multi-model gateway (vgpu serve) this
# is the alias to load/swap on the fly; for a fixed single-model server it is
# ignored by the backend. Defaults to "m" for backward compatibility.
model = "m"
for a in sys.argv[2:]:
    if a == "--no-stream":
        continue
    if a.startswith("--model="):
        model = a.split("=", 1)[1]
    elif a == "--model":
        pass  # value follows; handled below
    else:
        model = a  # positional <model> after port
# Handle "--model <value>" (flag then value) form.
for i, a in enumerate(sys.argv[2:]):
    if a == "--model" and i + 1 < len(sys.argv[2:]):
        model = sys.argv[2:][i + 1]
url = f"http://127.0.0.1:{port}/v1/chat/completions"
messages = []
print(cyan("vgpu") + dim("> type /exit to quit") +
      (dim("  (streaming on)") if not no_stream else dim("  (streaming off)")) +
      (dim(f"  (model: {model})") if model != "m" else ""))

def ask(hist):
    data = json.dumps({
        "model": model, "messages": hist,
        "temperature": 0.7, "max_tokens": 512, "stream": not no_stream,
    }).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=600)
    except Exception as e:
        print(red("error:") + " " + str(e)); return None
    if no_stream:
        try:
            obj = json.load(resp)
            text = obj["choices"][0]["message"]["content"]
        except Exception as e:
            print(red("error:") + " " + str(e)); return None
        print(text + "\n")
        return text
    # Streaming: parse Server-Sent Events, print delta.content as it arrives.
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
            sys.stdout.write(delta); sys.stdout.flush()
            full.append(delta)
    print()
    return "".join(full)

while True:
    try:
        line = input(prompt()).strip()
    except (EOFError, KeyboardInterrupt):
        break
    if line in ("/exit", "/quit"):
        break
    if not line:
        continue
    messages.append({"role": "user", "content": line})
    reply = ask(messages)
    if reply is None:
        messages.pop()
        continue
    messages.append({"role": "assistant", "content": reply})
