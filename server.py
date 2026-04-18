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

class Msg(BaseModel):
    message: str
    prefer_local: bool = False

LAUNCH = re.compile(r"^\s*(open|launch|start)\s+(\S+)", re.I)
SYSQ = re.compile(r"\b(disk|memory|battery|processes|uptime|wifi)\b", re.I)

def classify(m):
    if LAUNCH.match(m): return "launch"
    if SYSQ.search(m): return "system_query"
    if any(k in m.lower() for k in ("write ", "fix ", "debug ", "refactor ", "code ")): return "code"
    return "chat"

def run_launch(m):
    app_name = LAUNCH.match(m).group(2)
    subprocess.Popen([app_name], start_new_session=True)
    return f"launched {app_name}", 0

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
