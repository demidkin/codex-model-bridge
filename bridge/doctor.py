"""Read-only compatibility and authentication checks without credential extraction."""

import asyncio
import json
from .config import subscription_environment
from .rpc import MAX_LINE, stop_process


async def command(*args, env=None):
    """Capture a short diagnostic command with a bounded lifetime."""
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        env=env, start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 20)
        return process.returncode, (stdout + stderr).decode().strip()
    finally:
        await stop_process(process)


async def claude_capabilities(settings):
    """Request native Claude initialization metadata without sending a model prompt."""
    process = await asyncio.create_subprocess_exec(
        settings.claude, "-p", "--input-format", "stream-json", "--output-format",
        "stream-json", "--verbose", "--include-partial-messages",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL, env=subscription_environment(),
        limit=MAX_LINE, start_new_session=True,
    )
    try:
        process.stdin.write(b'{"type":"control_request","request_id":"bridge-init","request":{"subtype":"initialize"}}\n')
        await process.stdin.drain()
        async with asyncio.timeout(25):
            while line := await process.stdout.readline():
                data = json.loads(line)
                if data.get("type") == "control_response":
                    response = data["response"]
                    if response["subtype"] != "success":
                        raise ValueError("Claude initialization was rejected")
                    return response["response"]
        raise ValueError("Claude did not return capabilities")
    finally:
        await stop_process(process)


async def main(settings, arguments):
    """Print sanitized diagnostics and optionally refresh native Claude model metadata."""
    settings.prepare_state()
    checks = {}
    _, core_version = await command(settings.core, "--version")
    checks["core_version"] = core_version
    checks["core_version_tested"] = core_version == settings.tested_core_version
    code, login = await command(settings.core, "login", "status")
    checks["openai_chatgpt_subscription"] = code == 0 and "Logged in using ChatGPT" in login
    _, claude_version = await command(settings.claude, "--version")
    checks["claude_version"] = claude_version
    checks["claude_version_tested"] = claude_version == settings.tested_claude_version
    code, auth = await command(settings.claude, "auth", "status", env=subscription_environment())
    status = json.loads(auth) if code == 0 else {}
    checks["claude_auth"] = {key: status.get(key) for key in (
        "loggedIn", "authMethod", "apiProvider", "subscriptionType",
    )}
    try:
        settings.check_deepseek_key()
        checks["deepseek_key"] = "ready"
    except ValueError as exc:
        checks["deepseek_key"] = str(exc)
    checks["enabled"] = {"deepseek": settings.enable_deepseek, "claude": settings.enable_claude}
    if "--refresh-claude" in arguments:
        capabilities = await claude_capabilities(settings)
        path = settings.state / "claude-models.json"
        path.write_text(json.dumps(capabilities["models"], ensure_ascii=False, indent=2) + "\n")
        path.chmod(0o600)
        checks["claude_models"] = [model["value"] for model in capabilities["models"]]
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if checks["core_version_tested"] and checks["openai_chatgpt_subscription"] else 1
