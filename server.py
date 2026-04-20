"""Scatter router v1.1. Routes + IPW logging + web UI."""
import json, os, re, subprocess, time, urllib.request
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from anthropic import Anthropic

load_dotenv(Path(__file__).parent / ".env")
claude = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
OLLAMA = "http://127.0.0.1:11434/api/generate"
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "qwen2.5-coder:1.5b")

LOG_PATH = Path.home() / ".scatter" / "ipw-log.jsonl"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
WATTS = {"local:shell": 1.0, "local:launch": 0.5, "cloud:sonnet": 5.0, "local:qwen": 30.0}

def log_ipw(route, duration_s, tokens):
    entry = {
        "timestamp": time.time(),
        "route": route,
        "tokens": tokens,
        "duration_s": round(duration_s, 3),
        "watts_estimated": WATTS.get(route, 5.0),
        "watt_seconds": round(duration_s * WATTS.get(route, 5.0), 3),
        "note": "estimated_not_metered",
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")

app = FastAPI()

@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")

@app.get("/ipw-summary")
def ipw_summary():
    cutoff = time.time() - 86400
    tokens = 0
    watt_seconds = 0
    calls = 0
    if LOG_PATH.exists():
        with LOG_PATH.open() as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if e.get("timestamp", 0) > cutoff:
                        tokens += e.get("tokens", 0)
                        watt_seconds += e.get("watt_seconds", 0)
                        calls += 1
                except Exception:
                    pass
    return {"tokens": tokens, "watt_hours": watt_seconds / 3600, "calls": calls}

@app.get("/stats")
def stats():
    cutoff = time.time() - 86400
    routes = {}
    if LOG_PATH.exists():
        with LOG_PATH.open() as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if e.get("timestamp", 0) <= cutoff:
                        continue
                    r = routes.setdefault(e["route"], {"calls": 0, "tokens": 0, "watt_seconds": 0.0})
                    r["calls"] += 1
                    r["tokens"] += e.get("tokens", 0)
                    r["watt_seconds"] += e.get("watt_seconds", 0)
                except Exception:
                    pass
    out = []
    for name, r in routes.items():
        tok_per_ws = r["tokens"] / r["watt_seconds"] if r["watt_seconds"] > 0 else None
        out.append({
            "route": name,
            "calls": r["calls"],
            "tokens": r["tokens"],
            "watt_seconds": round(r["watt_seconds"], 3),
            "tokens_per_watt_second": round(tok_per_ws, 3) if tok_per_ws is not None else None,
        })
    out.sort(key=lambda x: (x["tokens_per_watt_second"] or 0), reverse=True)
    return {"window": "24h", "routes": out, "note": "watts_estimated_not_metered"}

class Msg(BaseModel):
    message: str
    prefer_local: bool = False

LAUNCH_VERB = re.compile(r"\b(open|launch|start|run|show|fire up|pull up|bring up)\b", re.I)
LAUNCH_TARGETS = {
    "firefox": "firefox", "browser": "firefox", "web": "firefox",
    "files": "nautilus", "file manager": "nautilus", "finder": "nautilus",
    "terminal": "gnome-terminal", "shell": "gnome-terminal", "console": "gnome-terminal",
}
SYSQ = re.compile(r"\b(disk|memory|battery|processes|uptime|wifi)\b", re.I)

def _launch_target(m):
    if not LAUNCH_VERB.search(m): return None
    lower = m.lower()
    for key in sorted(LAUNCH_TARGETS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", lower):
            return LAUNCH_TARGETS[key]
    return None

def classify(m):
    if _launch_target(m): return "launch"
    if SYSQ.search(m): return "system_query"
    if any(k in m.lower() for k in ("write ", "fix ", "debug ", "refactor ", "code ")): return "code"
    return "chat"

def run_launch(m):
    cmd = _launch_target(m)
    subprocess.Popen(cmd, shell=True, start_new_session=True)
    return f"launched {cmd}", 0

def run_system(m):
    out = subprocess.run(["bash","-c","uptime && free -h && df -h /"],
                         capture_output=True, text=True).stdout
    return out, 0

def run_local(m):
    req = urllib.request.Request(OLLAMA,
        data=json.dumps({"model": LOCAL_MODEL, "prompt": m, "stream": False}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    tokens = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
    return data["response"], tokens

def run_cloud(m):
    r = claude.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role":"user","content":m}])
    tokens = r.usage.input_tokens + r.usage.output_tokens
    return r.content[0].text, tokens

@app.post("/chat")
def chat(msg: Msg):
    t0 = time.time()
    intent = classify(msg.message)
    if intent == "launch":          (resp, tokens), route = run_launch(msg.message), "local:launch"
    elif intent == "system_query":  (resp, tokens), route = run_system(msg.message), "local:shell"
    elif intent == "code":          (resp, tokens), route = run_cloud(msg.message), "cloud:sonnet"
    elif msg.prefer_local:          (resp, tokens), route = run_local(msg.message), "local:qwen"
    else:                            (resp, tokens), route = run_cloud(msg.message), "cloud:sonnet"
    duration = time.time() - t0
    log_ipw(route, duration, tokens)
    return {"response": resp, "route": route, "tokens": tokens, "ms": int(duration * 1000)}
