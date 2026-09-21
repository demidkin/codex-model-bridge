"""Launch or restore the desktop without patching it or setting global environment."""

import argparse
import asyncio
import json
import os
import plistlib
import shutil
import subprocess
import time
from pathlib import Path
from .config import ROOT, Settings, subscription_environment
from .doctor import command


APP = Path("/Applications/ChatGPT.app")
TESTED_APP = ("26.915.31945", "9922")


def running(bundle_id):
    """Read running application identities; Electron can replace its process argv."""
    script = (
        'ObjC.import("AppKit"); '
        'const apps = $.NSWorkspace.sharedWorkspace.runningApplications; '
        'let active = false; for (let i = 0; i < apps.count; i++) { '
        'if (ObjC.unwrap(apps.objectAtIndex(i).bundleIdentifier) === '
        + json.dumps(bundle_id) + ') active = true; } JSON.stringify(active);'
    )
    result = subprocess.run(
        ["/usr/bin/osascript", "-l", "JavaScript", "-e", script],
        capture_output=True, text=True, timeout=10, check=True,
    )
    return json.loads(result.stdout)


async def launch(standard=False, check=False):
    """Prepare a verifiable launch; refuse to silently reuse an already-running app."""
    os.umask(0o077)
    settings = Settings.load()
    settings.prepare_state()
    info = plistlib.loads((APP / "Contents/Info.plist").read_bytes())
    app_version = (info["CFBundleShortVersionString"], info["CFBundleVersion"])
    active = running(info["CFBundleIdentifier"])
    _, core_version = await command(settings.core, "--version")
    if not standard and (app_version != TESTED_APP or core_version != settings.tested_core_version):
        raise ValueError("Версия Codex изменилась. Перед запуском адаптера нужна повторная проверка протокола.")
    if not standard and settings.enable_claude:
        code, version = await command(settings.claude, "--version")
        if code or version != settings.tested_claude_version:
            raise ValueError("Версия Claude Code изменилась: нужна проверка совместимости.")
        code, raw = await command(settings.claude, "auth", "status", env=subscription_environment())
        auth = json.loads(raw) if code == 0 else {}
        if not (auth.get("loggedIn") and auth.get("authMethod") == "claude.ai" and
                auth.get("apiProvider") == "firstParty" and auth.get("subscriptionType")):
            raise ValueError("Claude Code должен быть авторизован по подписке Claude.")
    if not standard and settings.enable_deepseek:
        settings.check_deepseek_key()
    if not standard:
        code, auth = await command(settings.core, "login", "status")
        if code or "Logged in using ChatGPT" not in auth:
            raise ValueError("Штатный Codex должен быть авторизован через ChatGPT.")
    arguments = ["/usr/bin/open", "-a", str(APP),
                 "--env", "CODEX_CLI_PATH=" + ("" if standard else str(ROOT / "bin/codex-bridge")),
                 "--env", "CODEX_APP_SERVER_FORCE_CLI=" + ("" if standard else "1")]
    plan = {"mode": "standard" if standard else "bridge", "app_running": active,
            "app_version": list(app_version), "core_version": core_version,
            "deepseek_enabled": settings.enable_deepseek, "claude_enabled": settings.enable_claude, "launch_command": arguments}
    if check:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if active:
        print("Сначала полностью закрой Codex через ⌘Q, затем запусти этот файл ещё раз.\n"
              "Уже запущенное приложение не подхватит новый способ запуска.")
        return 2
    if not standard:
        source = Path.home() / ".codex/config.toml"
        if source.exists():
            backup = settings.state / "backups" / str(time.time_ns())
            backup.mkdir(parents=True, mode=0o700)
            shutil.copyfile(source, backup / "config.toml")
            (backup / "config.toml").chmod(0o600)
    subprocess.run(arguments, check=True)
    (settings.state / "last-launch.json").write_text(json.dumps({**plan, "launched_at": int(time.time())}, indent=2) + "\n")
    print("Codex запущен штатно." if standard else "Codex запущен через адаптер. Переписка и подписка остаются в штатном клиенте.")
    return 0


def main():
    """Provide check-only, bridge, and standard launch commands."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard", action="store_true")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    try:
        return asyncio.run(launch(arguments.standard, arguments.check))
    except (ValueError, OSError) as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
