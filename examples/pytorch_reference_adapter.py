#!/usr/bin/env python3
"""Reference local adapter for the Intelligence Network workload protocol.

This script speaks the adapter JSON-line protocol used by the workload runtime.
It is intentionally framework-agnostic at the network layer: the core only knows
that a workload is mapped to an adapter name and a protocol version. The real ML
implementation remains in the external adapter process.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict

ADAPTER_NAME = "training.pytorch.v1"
ADAPTER_VERSION = "0.1.0"


def reply(request_id: int, ok: bool, status: str, payload: Any) -> None:
    payload_value = payload if isinstance(payload, (dict, list, str, int, float, bool)) or payload is None else str(payload)
    line = json.dumps(
        {
            "request_id": request_id,
            "ok": ok,
            "status": status,
            "payload": json.dumps(payload_value, separators=(",", ":")),
        },
        separators=(",", ":"),
    )
    print(line)
    sys.stdout.flush()


def handle_message(message: Any, request_id: int) -> None:
    if isinstance(message, str):
        variant = message
        payload = {}
    elif isinstance(message, dict):
        if len(message) == 1:
            variant, payload = next(iter(message.items()))
        else:
            reply(request_id, False, "invalid-message", {"error": "unexpected message shape", "message": message})
            return
    else:
        reply(request_id, False, "invalid-message", {"error": "message payload is not an object or string", "message": message})
        return

    if variant == "Hello":
        session = payload or {}
        reply(
            request_id,
            True,
            "ready",
            {
                "adapter": ADAPTER_NAME,
                "version": ADAPTER_VERSION,
                "framework": "pytorch",
                "capabilities": ["training", "inference", "checkpoint"],
                "hello": session,
            },
        )
        return

    if variant == "Probe":
        capability = (payload or {}).get("capability", "")
        reply(
            request_id,
            True,
            "ok",
            {"capability": capability, "supported": capability in {"training", "inference", "checkpoint", "text"}},
        )
        return

    if variant == "Prepare":
        manifest = (payload or {}).get("manifest", "")
        reply(
            request_id,
            True,
            "prepared",
            {"adapter": ADAPTER_NAME, "manifest": manifest, "framework": "pytorch", "status": "ready"},
        )
        return

    if variant == "LoadModel":
        artifact_id = (payload or {}).get("artifact_id", "")
        reply(
            request_id,
            True,
            "model-loaded",
            {"artifact_id": artifact_id, "adapter": ADAPTER_NAME, "framework": "pytorch"},
        )
        return

    if variant == "LoadShard":
        shard_id = (payload or {}).get("shard_id", "")
        reply(
            request_id,
            True,
            "shard-loaded",
            {"shard_id": shard_id, "adapter": ADAPTER_NAME, "framework": "pytorch"},
        )
        return

    if variant == "TrainWindow":
        request = (payload or {}).get("request", "")
        reply(
            request_id,
            True,
            "training-window-accepted",
            {"adapter": ADAPTER_NAME, "request": request, "framework": "pytorch", "steps": 1},
        )
        return

    if variant == "Checkpoint":
        generation = (payload or {}).get("generation", 0)
        reply(
            request_id,
            True,
            "checkpoint-created",
            {"adapter": ADAPTER_NAME, "generation": generation, "framework": "pytorch"},
        )
        return

    if variant == "Restore":
        checkpoint_id = (payload or {}).get("checkpoint_id", "")
        reply(
            request_id,
            True,
            "checkpoint-restored",
            {"adapter": ADAPTER_NAME, "checkpoint_id": checkpoint_id, "framework": "pytorch"},
        )
        return

    if variant == "Health":
        reply(
            request_id,
            True,
            "ok",
            {"adapter": ADAPTER_NAME, "version": ADAPTER_VERSION, "framework": "pytorch", "status": "healthy"},
        )
        return

    if variant == "Cancel":
        reply(request_id, True, "cancelled", {"adapter": ADAPTER_NAME, "framework": "pytorch"})
        return

    if variant == "Shutdown":
        reply(request_id, True, "shutdown", {"adapter": ADAPTER_NAME, "framework": "pytorch"})
        raise SystemExit(0)

    reply(request_id, False, "unsupported", {"adapter": ADAPTER_NAME, "message": message})


def main() -> None:
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            request = json.loads(line)
            request_id = int(request.get("request_id", 0))
            envelope = request.get("envelope", {})
            message = envelope.get("message", {})
            handle_message(message, request_id)
        except json.JSONDecodeError as exc:
            print(json.dumps({"request_id": 0, "ok": False, "status": "invalid-json", "payload": json.dumps({"error": str(exc)})}, separators=(",", ":")))
            sys.stdout.flush()
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
