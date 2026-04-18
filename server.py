
"""Scatter router. Minimal. Real."""
import json, os, re, subprocess, time, urllib.request
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from anthropic import Anthropic

load_dotenv(Path(__file__).parent / ".env")
claude = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
OLLAMA = "http://127.0.0.1:11434/api/generate"
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "qwen2.5-coder:1.5b")

app = FastAPI()

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
    return f"launched {app_name}"

def run_system(m):
    return subprocess.run(["bash","-c","uptime && free -h && df -h /"],
                          capture_output=True, text=True).stdout

def run_local(m):
    req = urllib.request.Request(OLLAMA,
        data=json.dumps({"model": LOCAL_MODEL, "prompt": m, "stream": False}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["response"]

def run_cloud(m):
    r = claude.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role":"user","content":m}])
    return r.content[0].text

@app.post("/chat")
def chat(msg: Msg):
    t0 = time.time()
    intent = classify(msg.message)
    if intent == "launch":          resp, route = run_launch(msg.message), "local:launch"
    elif intent == "system_query":  resp, route = run_system(msg.message), "local:shell"
    elif intent == "code":          resp, route = run_cloud(msg.message), "cloud:sonnet"
    elif msg.prefer_local:          resp, route = run_local(msg.message), "local:qwen"
    else:                            resp, route = run_cloud(msg.message), "cloud:sonnet"
    return {"response": resp, "route": route, "ms": int((time.time()-t0)*1000)}
