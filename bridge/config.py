"""Configuration and local secret references; never copies subscription tokens."""

import json
import os
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Explicit executables and private state belonging to this adapter."""

    core: str = "/Applications/ChatGPT.app/Contents/Resources/codex"
    claude: str = str(Path.home() / ".local/bin/claude")
    state: Path = ROOT / ".runtime"
    deepseek_key: Path = Path.home() / ".codex/deepseek.key"
    deepseek_catalog: Path = ROOT / "resources/deepseek-models.json"
    enable_deepseek: bool = False
    enable_claude: bool = False
    enable_agents: bool = True
    enable_skills: bool = True
    tested_core_version: str = "codex-cli 0.155.0-alpha.9.2"
    tested_claude_version: str = "2.1.272 (Claude Code)"

    @classmethod
    def load(cls):
        """Load a local TOML file without reading global Codex configuration."""
        path = Path(os.environ.get("CODEX_BRIDGE_CONFIG", ROOT / "bridge.local.toml"))
        values = tomllib.loads(path.read_text()) if path.exists() else {}
        options = values.get("bridge", {})
        for name in ("state", "deepseek_key", "deepseek_catalog"):
            if name in options:
                options[name] = Path(options[name]).expanduser().resolve()
        settings = cls(**options)
        if Path(settings.core).resolve() == (ROOT / "bin/codex-bridge").resolve():
            raise ValueError("The original Codex executable must not be the adapter")
        return settings

    def prepare_state(self):
        """Create a private state directory; reject directories owned by others."""
        self.state.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.state.is_symlink() or self.state.stat().st_uid != os.getuid():
            raise ValueError("Adapter state directory has unsafe ownership")
        self.state.chmod(0o700)

    def check_deepseek_key(self):
        """Validate a regular private key file without loading its contents."""
        try:
            info = self.deepseek_key.lstat()
        except FileNotFoundError as exc:
            raise ValueError(f"DeepSeek key is missing: {self.deepseek_key}") from exc
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("DeepSeek key must be a regular file owned by this user")
        if stat.S_IMODE(info.st_mode) != 0o600 or info.st_size == 0:
            raise ValueError("DeepSeek key must be nonempty with permissions 0600")

    def provider_config(self, engine):
        """Return thread-local overrides, leaving the OpenAI defaults untouched."""
        if engine == "deepseek":
            return {
                "model_providers.bridge_deepseek": {
                    "name": "DeepSeek",
                    "base_url": "https://api.deepseek.com",
                    "wire_api": "responses",
                    "auth": {"command": "/bin/cat", "args": [str(self.deepseek_key)]},
                },
                "model_catalog_json": str(self.deepseek_catalog),
                "web_search": "disabled",
                "model_reasoning_summary": "none",
            }
        if engine == "claude":
            # Core owns task metadata only. Never an inference endpoint for Claude.
            return {
                "model_providers.bridge_claude": {
                    "name": "Claude Code (metadata only)",
                    "base_url": "http://127.0.0.1:9",
                    "wire_api": "responses",
                    "request_max_retries": 0,
                },
                "web_search": "disabled",
            }
        return {}


def subscription_environment():
    """Remove alternate Claude billing routes from only the launched child."""
    env = dict(os.environ)
    for name in (
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY", "CLAUDE_CODE_SIMPLE", "CLAUDECODE",
    ):
        env.pop(name, None)
    return env


def read_models(path):
    """Read a provider catalogue, whose contents are data rather than executable code."""
    document = json.loads(Path(path).read_text())
    models = document.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("Invalid model catalogue")
    return models
