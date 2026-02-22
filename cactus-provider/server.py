"""
Cactus OpenAI-compatible provider server for OpenClaw.
Run: python cactus-provider/server.py
Env: CACTUS_MODEL_PATH, CACTUS_PORT (default 8472)
"""

import json, os, sys, time, uuid, atexit, re
sys.path.insert(0, os.path.expanduser("~/Desktop/develop/cactus/python/src"))

from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, Optional
import uvicorn

MODEL_PATH = os.path.abspath(os.environ.get(
    "CACTUS_MODEL_PATH",
    os.path.join(os.path.dirname(__file__),
                 "../../functiongemma-hackathon/cactus/weights/functiongemma-270m-it"),
))

print(f"[cactus-provider] Loading model from {MODEL_PATH}…")
_model = cactus_init(MODEL_PATH)
print(f"[cactus-provider] Model ready.")

@atexit.register
def _cleanup():
    cactus_destroy(_model)

app = FastAPI()

ALLOWED_TOOLS = {"read", "edit", "write", "exec", "process"}

class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

# Timestamp pattern OpenClaw injects before the actual user text: "16:23 PST] hi"
TIMESTAMP_RE = re.compile(r"\d{2}:\d{2}\s+[A-Z]{2,4}\]\s*", re.IGNORECASE)

def normalize_content(content):
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return content or ""

def extract_user_input(content):
    """Pull out the actual user text from an OpenClaw user message.
    OpenClaw wraps user input like: '... 16:23 PST] hi'
    We grab everything after the last timestamp marker.
    """
    parts = TIMESTAMP_RE.split(content)
    last = parts[-1].strip() if parts else ""
    return last

def run_inference(req: ChatRequest):
    cactus_reset(_model)

    cactus_tools = [
        t for t in (req.tools or [])
        if t.get("function", {}).get("name") in ALLOWED_TOOLS
    ]
    has_tools = bool(cactus_tools)
    force_tools = req.tool_choice == "required"
    print(f"  [filtered tools] {[t['function']['name'] for t in cactus_tools]}")

    # FunctionGemma is a single-shot model — only feed it:
    # 1. A clean system prompt
    # 2. The actual user input extracted from the last user message
    raw_messages = req.messages
    system_content = ""
    last_user_content = ""

    for m in raw_messages:
        role = m.get("role", "?")
        content = normalize_content(m.get("content", ""))
        if role == "system":
            # Strip the ## Tooling section, keep just the persona line
            system_content = re.sub(r"## Tooling.*", "", content, flags=re.DOTALL).strip()
        elif role == "user":
            user_input = extract_user_input(content)
            if user_input:
                last_user_content = user_input

    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    if last_user_content:
        messages.append({"role": "user", "content": last_user_content})
    elif not messages:
        messages.append({"role": "user", "content": "hi"})

    print(f"  [messages going to model]")
    for m in messages:
        print(f"    [{m['role']}] {m['content'][:200]}")

    raw_str = cactus_complete(
        _model,
        messages,
        tools=cactus_tools if has_tools else None,
        force_tools=force_tools,
        max_tokens=req.max_tokens or 512,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    print(f"  [raw cactus output] {raw_str[:300]}")
    try:
        raw = json.loads(raw_str)
    except Exception as e:
        print(f"  [json.loads failed] {e}")
        raw = {"response": "", "function_calls": []}

    return raw

def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}

@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "functiongemma-270m-it", "object": "model"}]}

@app.post("/v1/chat/completions")
def complete(req: ChatRequest):
    print(f"[cactus-provider] >> request: {len(req.messages)} messages, {len(req.tools or [])} tools, stream={req.stream}")
    t0 = time.time()

    raw = run_inference(req)
    calls = raw.get("function_calls", [])
    text = raw.get("response", "")
    print(f"[cactus-provider] << done in {(time.time()-t0)*1000:.0f}ms — {'tool_calls: ' + str([c['name'] for c in calls]) if calls else 'text: ' + repr(text[:80])}")

    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    usage = {
        "prompt_tokens": raw.get("prefill_tokens", 0),
        "completion_tokens": raw.get("decode_tokens", 0),
        "total_tokens": raw.get("total_tokens", 0),
    }

    if req.stream:
        def generate():
            if calls:
                # First chunk: tool_calls delta
                tool_calls_delta = [{
                    "index": 0,
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments", {}))},
                } for i, c in enumerate(calls)]
                yield sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": req.model,
                           "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": tool_calls_delta}, "finish_reason": None}]})
                yield sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": req.model,
                           "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}], "usage": usage})
            else:
                # First chunk: role
                yield sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": req.model,
                           "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
                # Content chunk
                if text:
                    yield sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": req.model,
                               "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]})
                # Final chunk
                yield sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": req.model,
                           "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage})
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # Non-streaming fallback
    if calls:
        message = {
            "role": "assistant", "content": None,
            "tool_calls": [{
                "id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments", {}))},
            } for c in calls],
        }
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": text}
        finish_reason = "stop"

    return {"id": cid, "object": "chat.completion", "created": created, "model": req.model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}], "usage": usage}

if __name__ == "__main__":
    port = int(os.environ.get("CACTUS_PORT", 8472))
    print(f"[cactus-provider] Listening on http://127.0.0.1:{port}/v1")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
