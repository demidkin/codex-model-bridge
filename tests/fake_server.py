"""Isolated transport fixture; deliberately unrelated to model APIs."""

import json
import sys


def send(message):
    """Flush a complete response to a potentially blocked parent."""
    print(json.dumps(message, ensure_ascii=False), flush=True)


pending = []
for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if method == "echo":
        send({"id": message["id"], "result": message["params"]})
    elif method == "reverse":
        pending.append(message)
        if len(pending) == 2:
            for item in reversed(pending):
                send({"id": item["id"], "result": item["params"]})
            pending.clear()
    elif method == "approval":
        pending.append(message)
        send({"id": 17, "method": "item/commandExecution/requestApproval", "params": {"marker": "fixture"}})
    elif message.get("id") == 17:
        send({"id": pending.pop()["id"], "result": message["result"]})
    elif method == "crash":
        sys.exit(23)
