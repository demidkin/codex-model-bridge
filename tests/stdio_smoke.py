"""Read-only native smoke check through the installed executable wrapper."""

import asyncio
import json
from pathlib import Path
from bridge.rpc import CoreClient


async def main():
    """Initialize the actual wrapper, list models, and verify clean shutdown."""
    events = []

    async def event(message):
        events.append(message.get("method"))

    launcher = Path(__file__).resolve().parent.parent / "bin/codex-bridge"
    client = CoreClient([str(launcher), "app-server"], event)
    await client.start()
    try:
        async with asyncio.timeout(20):
            response = await client.call("initialize", {
                "clientInfo": {"name": "bridge_stdio_smoke", "version": "0.1.0"},
            })
            await client.send({"method": "initialized"})
            models = await client.call("model/list", {"includeHidden": True, "limit": 100})
            config = await client.call("config/read", {})
    finally:
        await client.close()
    report = {
        "initialize": "ok", "platform": response["platformOs"],
        "models": [model["model"] for model in models["data"]],
        "wrapper_exit_code": client.process.returncode,
        "desktop_catalog_hint": bool(config["config"].get("model_catalog_json")),
    }
    assert "gpt-6-astra" in report["models"]
    assert report["wrapper_exit_code"] == 0
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
