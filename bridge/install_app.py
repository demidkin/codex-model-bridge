"""Build a Finder-launchable macOS app using the system AppleScript compiler."""

import json
import os
import plistlib
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from .config import ROOT, Settings


APP_NAME = "Codex Bridge.app"
BUNDLE_ID = "local.codex.model-bridge.launcher"


def build_app():
    """Create and locally sign our launcher; preserve any previous generated app."""
    os.umask(0o077)
    settings = Settings.load()
    settings.prepare_state()
    destination = ROOT / "bin" / APP_NAME
    marker_path = Path("Contents/Resources/bridge-launcher.json")
    if destination.exists():
        marker = destination / marker_path
        if not marker.is_file() or json.loads(marker.read_text()).get("generator") != "codex-model-bridge":
            raise ValueError("An unrecognized application already exists at the destination")
    command = shlex.join([sys.executable, str(ROOT / "run_bridge.py"), "--desktop-launch"])
    report_path = settings.state / "app-launcher-check.json"
    check = command + " --check > " + shlex.quote(str(report_path))
    source = (ROOT / "resources/launcher.applescript.in").read_text()
    source = source.replace("__LAUNCH_COMMAND__", json.dumps(command, ensure_ascii=False))
    source = source.replace("__CHECK_COMMAND__", json.dumps(check, ensure_ascii=False))
    with tempfile.TemporaryDirectory(prefix="app-build-", dir=settings.state) as temporary:
        staging = Path(temporary)
        script = staging / "launcher.applescript"
        script.write_text(source)
        built = staging / APP_NAME
        subprocess.run(["/usr/bin/osacompile", "-o", str(built), str(script)], check=True)
        info_path = built / "Contents/Info.plist"
        info = plistlib.loads(info_path.read_bytes())
        info.update({
            "CFBundleIdentifier": BUNDLE_ID,
            "CFBundleName": "Codex Bridge",
            "CFBundleDisplayName": "Codex Bridge",
            "CFBundleShortVersionString": "0.1.0",
            "CFBundleVersion": "1",
            "LSUIElement": True,
            "NSHighResolutionCapable": True,
        })
        info_path.write_bytes(plistlib.dumps(info))
        (built / marker_path).write_text(json.dumps({
            "generator": "codex-model-bridge", "source_root": str(ROOT), "format": 1,
        }, indent=2) + "\n")
        subprocess.run(["/usr/bin/codesign", "--force", "--sign", "-", str(built)], check=True)
        subprocess.run(["/usr/bin/codesign", "--verify", "--strict", "--deep", str(built)], check=True)
        if destination.exists():
            backup = settings.state / "app-backups" / str(time.time_ns())
            backup.mkdir(parents=True, mode=0o700)
            destination.rename(backup / APP_NAME)
        destination.parent.mkdir(exist_ok=True)
        try:
            built.rename(destination)
        except OSError:
            if "backup" in locals():
                (backup / APP_NAME).rename(destination)
            raise
    print(destination)
    return destination


if __name__ == "__main__":
    try:
        build_app()
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"App launcher build failed: {error}", file=sys.stderr)
        raise SystemExit(1)
