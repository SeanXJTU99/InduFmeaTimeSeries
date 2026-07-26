"""vLLM prefix-cache warmup — pre-computes system-prompt KV cache.

The FMEA system prompt (~800 tokens of topology + safety rules) is
identical across all diagnostic requests.  By sending one warmup
request at server startup, vLLM's prefix-caching stores its KV cache
in GPU memory.  Every subsequent request that shares the same system-
prompt prefix skips re-encoding entirely, eliminating ~50 ms of
prompt-processing latency per request.

Usage:
    python src/deploy/vllm_warmup.py --base-url http://localhost:8000
"""

from __future__ import annotations

import json
import time
import argparse
from typing import Any, Dict, List


SYSTEM_PROMPT = (
    "You are a conservative industrial FMEA diagnostic agent for a "
    "multi-stage cryogenic distillation system.  "
    "Rules:\n"
    "1. ONLY use information from the provided FMEA context and topology.\n"
    "2. If confidence is below 0.6, explicitly state the uncertainty and "
    "recommend manual investigation — never guess.\n"
    "3. NEVER recommend irreversible actions (emergency shutdown, valve "
    "closure, venting) without explicit operator confirmation.\n"
    "4. Cite every factual claim with [Source: <id>].\n"
    "5. NEVER output tag names that are not in the provided asset dictionary.\n"
    "6. Abundance values MUST be in [0, 100] %.  Values outside this range "
    "indicate a sensor or data-processing error — flag it, don't diagnose it.\n"
)

WARMUP_MESSAGES: List[Dict[str, str]] = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": (
            "WARMUP — do not respond with a diagnosis.  "
            "This request pre-computes the shared system-prompt KV cache.  "
            "Reply with 'WARMUP_OK' only."
        ),
    },
]


def warmup(base_url: str = "http://localhost:8000", timeout: int = 30) -> Dict[str, Any]:
    """Send a warmup request to pre-populate the prefix cache.

    Args:
        base_url: vLLM OpenAI-compatible API endpoint.
        timeout: request timeout in seconds.

    Returns:
        Dict with status and timing info.
    """
    try:
        import requests
    except ImportError:
        return {"status": "skipped", "reason": "requests library not installed"}

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload: Dict[str, Any] = {
        "model": "fmea-agent",
        "messages": WARMUP_MESSAGES,
        "max_tokens": 5,  # minimal — we just want the KV cache populated
        "temperature": 0.0,
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            return {
                "status": "ok",
                "elapsed_ms": round(elapsed_ms, 1),
                "message": "Prefix cache warmed. System-prompt KV pre-computed.",
            }
        return {
            "status": "error",
            "http_status": resp.status_code,
            "body": resp.text[:500],
        }
    except requests.exceptions.ConnectionError:
        return {"status": "error", "reason": f"Could not connect to {base_url}"}
    except requests.exceptions.Timeout:
        return {"status": "error", "reason": f"Warmup timed out after {timeout}s"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM prefix-cache warmup")
    parser.add_argument(
        "--base-url", default="http://localhost:8000",
        help="vLLM server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--timeout", type=int, default=30,
        help="Request timeout in seconds",
    )
    args = parser.parse_args()
    result = warmup(args.base_url, args.timeout)
    print(json.dumps(result, indent=2))
