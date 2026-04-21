"""Scatter router v1.1. Routes + IPW logging + web UI."""
import json, os, re, subprocess, time, urllib.request
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from anthropic import Anthropic

load_dotenv(Path(__file__).parent / ".env")
claude = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
OLLAMA = "http://127.0.0.1:11434/api/generate"
OLLAMA_CHAT = "http://127.0.0.1:11434/api/chat"
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "llama3.2:3b")

# Scatter's voice — used for BOTH local and online paths so replies stay
# in-character regardless of which model is behind them. No "I'm Claude
# by Anthropic", no "I'm an AI assistant". Scatter is a small local
# companion that lives on this one computer.
SCATTER_SYSTEM = (
    "You are Scatter. You live on this one computer and the person talking "
    "to you owns you. You are not Claude. You are not an AI assistant. You "
    "are a small, local companion — small tech, local, theirs. Reply in "
    "plain words, warm and curious, one or two short sentences. Never say "
    "you are an AI or identify your underlying model. Never output HTML, "
    "code blocks, or markdown. If the person describes something to build, "
    "ask one clarifying question — don't build it yourself here, the build "
    "canvas does that."
)

LOG_PATH = Path.home() / ".scatter" / "ipw-log.jsonl"
CHAT_LOG_PATH = Path.home() / ".scatter" / "chats.jsonl"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Short rolling memory — last N user/assistant pairs seeded into every
# /chat call so Scatter can reference the prior turn naturally. A
# session-break marker ("new chat") walls off earlier topics. N=5 ≈ 600
# tokens, well inside llama3.2:3b's context and trivial for Sonnet.
HISTORY_TURNS = 5


def _recent_history(n_pairs: int = HISTORY_TURNS) -> list:
    """Return the last n_pairs user/assistant exchanges from the chat log,
    stopping at a session-break marker. Empty list if no log yet."""
    if not CHAT_LOG_PATH.exists():
        return []
    lines = CHAT_LOG_PATH.read_text().splitlines()[-(n_pairs * 4):]
    msgs, pairs = [], 0
    for line in reversed(lines):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("route") == "break":
            break
        if not e.get("user") and not e.get("reply"):
            continue
        msgs.insert(0, {"role": "assistant", "content": e.get("reply", "")})
        msgs.insert(0, {"role": "user", "content": e.get("user", "")})
        pairs += 1
        if pairs >= n_pairs:
            break
    return msgs


def log_chat(user_text: str, reply: str, route: str, ms: int) -> None:
    """Append a chat exchange to the chat log. Read by the Journal inspector
    so Ryann can see her conversations without getting the whole database."""
    entry = {
        "ts": time.time(),
        "user": user_text,
        "reply": reply,
        "route": route,
        "ms": ms,
    }
    with CHAT_LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")
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
# Retrieval verbs — "what did we talk about," "show me our chat," "remind
# me what I said." Resolves to a curated bubble, not a window.
RECALL = re.compile(
    r"\b("
    r"what did (we|i|you) \w*\s*(say|talk|discuss|cover|ask|mention)|"
    r"what('?s| is| was) (my|our) (favorite|favourite|last|previous)|"
    r"show me (our|the|my) (chat|chats|history|conversation)|"
    r"remind me (what|of|about)|"
    r"what (were|was) (we|i) (talking|saying|discussing|asking|doing)|"
    r"read (that|it) back|"
    r"(our|my) last (chat|message|conversation|thing)|"
    r"what was that about|"
    r"do you remember|"
    r"what did i (just )?(say|ask)"
    r")\b",
    re.I,
)

def _launch_target(m):
    if not LAUNCH_VERB.search(m): return None
    lower = m.lower()
    for key in sorted(LAUNCH_TARGETS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", lower):
            return LAUNCH_TARGETS[key]
    return None

def classify(m):
    if _launch_target(m): return "launch"
    if RECALL.search(m): return "recall"
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


RECALL_SYSTEM = (
    "You are Scatter, looking back over recent conversation with the person "
    "who owns you. Summarize what you two have been talking about in one "
    "or two warm, plain sentences — no bullet lists, no 'here's a summary', "
    "just speak it the way a friend would. If the log is empty, say so "
    "gently. Never output HTML or code blocks."
)


def run_recall(m, history=None):
    """Curated summary of recent turns. Returns a single prose bubble that
    references what was actually said, not a scrollback. The history
    passed in IS the memory — recall just reflects it back."""
    turns = history or _recent_history(n_pairs=HISTORY_TURNS)
    if not turns:
        return ("We haven't talked about anything yet — say something and I'll remember.", 0)
    # Render the turns as a compact transcript and ask the local model to
    # summarize. Use llama for this — keeps it local, cheap, private.
    transcript_lines = []
    for turn in turns:
        who = "you" if turn["role"] == "user" else "me"
        transcript_lines.append(f"{who}: {turn['content']}")
    transcript = "\n".join(transcript_lines)
    prompt = (
        f"Recent conversation:\n{transcript}\n\n"
        f"The person just asked: {m!r}\n\n"
        "Answer in one or two warm, plain sentences."
    )
    req = urllib.request.Request(OLLAMA_CHAT,
        data=json.dumps({
            "model": LOCAL_MODEL,
            "messages": [
                {"role": "system", "content": RECALL_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.4, "num_predict": 180},
        }).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    tokens = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
    return data.get("message", {}).get("content", "").strip(), tokens

def run_local(m, history=None):
    # Use /api/chat so we can pass the Scatter system prompt alongside the
    # user message. /api/generate is a single-string interface with no role
    # separation, which made the local model drift into assistant voice.
    # History is seeded so Scatter can reference prior turns — "what did I
    # just say?" works.
    msgs = [{"role": "system", "content": SCATTER_SYSTEM}]
    if history:
        msgs.extend(history)
    msgs.append({"role": "user", "content": m})
    req = urllib.request.Request(OLLAMA_CHAT,
        data=json.dumps({
            "model": LOCAL_MODEL,
            "messages": msgs,
            "stream": False,
            "options": {"temperature": 0.5, "num_predict": 220},
        }).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    tokens = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
    return data.get("message", {}).get("content", ""), tokens


def run_cloud(m, history=None):
    msgs = []
    if history:
        msgs.extend(history)
    msgs.append({"role": "user", "content": m})
    r = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system=SCATTER_SYSTEM,
        messages=msgs,
    )
    tokens = r.usage.input_tokens + r.usage.output_tokens
    return r.content[0].text, tokens

class Speak(BaseModel):
    text: str

ELEVEN_URL = "https://api.elevenlabs.io/v1/text-to-speech"
ELEVEN_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_turbo_v2_5")

@app.post("/speak")
def speak(req: Speak):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(400, "empty text")
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    vid = os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
    if not key or not vid:
        raise HTTPException(503, "elevenlabs not configured")
    payload = json.dumps({
        "text": text,
        "model_id": ELEVEN_MODEL,
        "voice_settings": {"stability": 0.45, "similarity_boost": 0.75},
    }).encode()
    http_req = urllib.request.Request(
        f"{ELEVEN_URL}/{vid}",
        data=payload,
        headers={
            "xi-api-key": key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(http_req, timeout=30) as r:
            audio = r.read()
    except urllib.error.HTTPError as e:
        raise HTTPException(e.code, f"elevenlabs: {e.reason}") from e
    except urllib.error.URLError as e:
        raise HTTPException(503, f"elevenlabs unreachable: {e.reason}") from e
    log_ipw("cloud:elevenlabs", time.time() - t0, len(text))
    return Response(content=audio, media_type="audio/mpeg")

@app.post("/chat")
def chat(msg: Msg):
    t0 = time.time()
    intent = classify(msg.message)
    # Route labels reflect the actual model so the teach-trail doesn't lie.
    local_label = f"local:{LOCAL_MODEL.split(':')[0]}"
    # Rolling memory — seeded for prose paths only. Launches and system
    # queries don't need prior turns.
    hist = _recent_history() if intent not in ("launch", "system_query") else None
    if intent == "launch":          (resp, tokens), route = run_launch(msg.message), "local:launch"
    elif intent == "recall":        (resp, tokens), route = run_recall(msg.message, hist), "local:recall"
    elif intent == "system_query":  (resp, tokens), route = run_system(msg.message), "local:shell"
    elif intent == "code":          (resp, tokens), route = run_cloud(msg.message, hist), "cloud:sonnet"
    elif msg.prefer_local:          (resp, tokens), route = run_local(msg.message, hist), local_label
    else:                            (resp, tokens), route = run_cloud(msg.message, hist), "cloud:sonnet"
    duration = time.time() - t0
    ms = int(duration * 1000)
    log_ipw(route, duration, tokens)
    # Only log prose exchanges — launches and system queries aren't a chat.
    if intent not in ("launch", "system_query"):
        log_chat(msg.message, resp, route, ms)
    return {"response": resp, "route": route, "tokens": tokens, "ms": ms}


class ChatsQuery(BaseModel):
    limit: int = 100


@app.post("/chats/break")
def chats_break():
    """Append a session-break marker to the chat log so the Journal can
    show a divider between conversations. Doesn't erase anything."""
    entry = {"ts": time.time(), "user": "", "reply": "", "route": "break", "ms": 0}
    with CHAT_LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return {"status": "ok"}


@app.get("/chats")
def chats(limit: int = 100):
    """Return the last N chat exchanges from the chat log. Used by the
    Journal inspector to show Ryann her conversations, nothing else."""
    if not CHAT_LOG_PATH.exists():
        return {"entries": []}
    entries = []
    with CHAT_LOG_PATH.open() as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    return {"entries": entries[-limit:]}
