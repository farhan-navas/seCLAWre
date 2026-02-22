"""
Cactus OpenAI-compatible provider server for OpenClaw.
Run: python cactus-provider/server.py
Env: CACTUS_MODEL_PATH, CACTUS_PORT (default 8472)
"""

import json, os, sys, time, uuid, atexit
sys.path.insert(0, os.path.expanduser("~/Desktop/develop/cactus/python/src"))

from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from fastapi import FastAPI
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

class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}

@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "functiongemma-270m-it", "object": "model"}]}

@app.post("/v1/chat/completions")
def complete(req: ChatRequest):
    print(f"[cactus-provider] >> request: {len(req.messages)} messages, {len(req.tools or [])} tools")
    t0 = time.time()

    cactus_reset(_model)

    ALLOWED_TOOLS = {"read", "edit", "write", "exec", "process"}
    cactus_tools = [
        t for t in (req.tools or [])
        if t.get("function", {}).get("name") in ALLOWED_TOOLS
    ]
    has_tools = bool(cactus_tools)
    force_tools = has_tools and req.tool_choice != "none"
    print(f"  [filtered tools] {[t['function']['name'] for t in cactus_tools]}")

    # Normalize messages:
    # 1. Extract plain text from content blocks ([{'type':'text','text':'...'}])
    # 2. Strip the ## Tooling section from system prompt (already handled via tools param)
    def normalize_content(content):
        if isinstance(content, list):
            return " ".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return content or ""

    def strip_tooling_section(text):
        import re
        return re.sub(r"## Tooling.*", "", text, flags=re.DOTALL).strip()

    messages = []
    for m in req.messages:
        role = m.get("role", "?")
        content = normalize_content(m.get("content", ""))
        if role == "system":
            content = strip_tooling_section(content)
        if content.strip():
            messages.append({"role": role, "content": content})

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

    try:
        raw = json.loads(raw_str)
    except Exception:
        raw = {"response": str(raw_str), "function_calls": []}

    calls = raw.get("function_calls", [])
    print(f"[cactus-provider] << done in {(time.time()-t0)*1000:.0f}ms — {'tool_calls: ' + str([c['name'] for c in calls]) if calls else 'text: ' + repr(raw.get('response','')[:80])}")

    if calls:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments", {}))},
            } for c in calls],
        }
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": raw.get("response", "")}
        finish_reason = "stop"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": raw.get("prefill_tokens", 0),
            "completion_tokens": raw.get("decode_tokens", 0),
            "total_tokens": raw.get("total_tokens", 0),
        },
    }

if __name__ == "__main__":
    port = int(os.environ.get("CACTUS_PORT", 8472))
    print(f"[cactus-provider] Listening on http://127.0.0.1:{port}/v1")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
