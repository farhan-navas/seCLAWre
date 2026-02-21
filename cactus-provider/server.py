"""
Cactus OpenAI-compatible provider server for OpenClaw.

Wraps FunctionGemma (on-device via Cactus) to expose a /v1/chat/completions
endpoint. OpenClaw treats this exactly like any other OpenAI-compatible provider
and handles all tool execution (file reads, curl, exec, etc.) on its side —
this server just does the LLM inference and returns tool_calls in OpenAI format.

Run:
    python cactus-provider/server.py

Env vars:
    CACTUS_MODEL_PATH  — path to model weights dir (default: functiongemma-270m-it)
    CACTUS_PORT        — port to listen on (default: 8472)
"""

import atexit
import json
import os
import sys
import threading
import time
import uuid

# Ensure cactus is importable whether or not the venv is activated
sys.path.insert(0, os.path.expanduser("~/Desktop/develop/cactus/python/src"))

from cactus import cactus_complete, cactus_destroy, cactus_init, cactus_reset
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Optional
import uvicorn

MODEL_PATH = os.environ.get(
    "CACTUS_MODEL_PATH",
    os.path.join(
        os.path.dirname(__file__),
        "../../functiongemma-hackathon/cactus/weights/functiongemma-270m-it",
    ),
)
MODEL_PATH = os.path.abspath(MODEL_PATH)

# ---------------------------------------------------------------------------
# Thread-local model pool — same pattern as functiongemma-hackathon/main.py.
# Each thread gets one persistent model instance; avoids init/destroy overhead.
# ---------------------------------------------------------------------------
_thread_local = threading.local()
_all_models: list = []
_all_models_lock = threading.Lock()


def _get_model():
    if not hasattr(_thread_local, "model"):
        print(f"[cactus-provider] Loading model on thread {threading.current_thread().name}…")
        _thread_local.model = cactus_init(MODEL_PATH)
        with _all_models_lock:
            _all_models.append(_thread_local.model)
    return _thread_local.model


@atexit.register
def _destroy_all():
    with _all_models_lock:
        for m in _all_models:
            try:
                cactus_destroy(m)
            except Exception:
                pass
        _all_models.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Cactus Provider", version="1.0.0")


class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = None
    stream: Optional[bool] = False


@app.get("/v1/models")
def list_models():
    """OpenAI-compatible model listing."""
    return {
        "object": "list",
        "data": [
            {
                "id": "functiongemma-270m-it",
                "object": "model",
                "created": 0,
                "owned_by": "cactus",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    """
    OpenAI-compatible chat completions endpoint.

    OpenClaw sends messages + tool schemas here. We run inference on-device
    via FunctionGemma and return either a text response or tool_calls.
    OpenClaw then executes the actual tools (file read, exec, curl, etc.)
    and sends the results back as tool result messages for the next turn.
    """
    model = _get_model()
    cactus_reset(model)  # clear KV cache from previous call

    # OpenAI tools format is [{"type": "function", "function": {...}}]
    # Cactus expects the same shape — pass through directly.
    cactus_tools = req.tools or []
    has_tools = bool(cactus_tools)

    # Only force tool use if caller hasn't explicitly set tool_choice to "none"
    force_tools = has_tools and req.tool_choice != "none"

    raw_str = cactus_complete(
        model,
        req.messages,
        tools=cactus_tools if has_tools else None,
        force_tools=force_tools,
        max_tokens=req.max_tokens or 512,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    try:
        raw = json.loads(raw_str)
    except (json.JSONDecodeError, TypeError):
        raw = {"response": str(raw_str), "function_calls": []}

    function_calls = raw.get("function_calls", [])
    text = raw.get("response", "")

    # Build OpenAI-format response
    if function_calls:
        tool_calls = [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": fc["name"],
                    # OpenAI expects arguments as a JSON string, not a dict
                    "arguments": json.dumps(fc.get("arguments", {})),
                },
            }
            for fc in function_calls
        ]
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        }
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": text}
        finish_reason = "stop"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": raw.get("prefill_tokens", 0),
            "completion_tokens": raw.get("decode_tokens", 0),
            "total_tokens": raw.get("total_tokens", 0),
        },
        # Pass through cactus metadata for debugging
        "_cactus": {
            "confidence": raw.get("confidence"),
            "time_to_first_token_ms": raw.get("time_to_first_token_ms"),
            "total_time_ms": raw.get("total_time_ms"),
            "decode_tps": raw.get("decode_tps"),
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


if __name__ == "__main__":
    port = int(os.environ.get("CACTUS_PORT", 8472))
    print(f"[cactus-provider] Model path: {MODEL_PATH}")
    print(f"[cactus-provider] Eagerly loading model…")
    _get_model()
    print(f"[cactus-provider] Ready — http://127.0.0.1:{port}/v1")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
