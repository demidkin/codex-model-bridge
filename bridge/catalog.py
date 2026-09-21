"""Map menu choices to concrete execution engines and provider capabilities."""

import json
import os
from pathlib import Path
from .config import read_models


PROVIDERS = {"openai": "openai", "deepseek": "bridge_deepseek", "claude": "bridge_claude"}


class Catalog:
    """A union of native OpenAI models and explicitly enabled external models."""

    def __init__(self, settings):
        self.settings = settings
        self.extra = []
        self.engines = {}
        if settings.enable_deepseek:
            for model in read_models(settings.deepseek_catalog):
                slug = model["slug"]
                self.engines[slug] = "deepseek"
                self.extra.append({
                    "id": slug, "model": slug,
                    "displayName": model["display_name"] + " · DeepSeek API",
                    "description": model["description"], "hidden": False,
                    "isDefault": False,
                    "defaultReasoningEffort": model["default_reasoning_level"],
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": x["effort"], "description": x["description"]}
                        for x in model["supported_reasoning_levels"]
                    ],
                    "inputModalities": model.get("input_modalities", ["text"]),
                })
        if settings.enable_claude:
            path = settings.state / "claude-models.json"
            if not path.exists():
                raise ValueError("Run bridge doctor --refresh-claude before enabling Claude")
            for model in json.loads(path.read_text()):
                slug = "claude-code/" + model["value"]
                self.engines[slug] = "claude"
                levels = model.get("supportedEffortLevels", ["medium"])
                self.extra.append({
                    "id": slug, "model": slug,
                    "displayName": model["displayName"] + " · Claude Code",
                    "description": model["description"], "hidden": False,
                    "isDefault": False,
                    "defaultReasoningEffort": "high" if "high" in levels else levels[0],
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": value, "description": value} for value in levels
                    ],
                    "inputModalities": ["text", "image"],
                })

    def ui_catalog_path(self):
        """Expose a local catalog hint to the desktop subscription model filter.

        This is never installed into native global config or OpenAI inference.
        Only model metadata is copied from the native cache, not account identity.
        """
        path = self.settings.state / 'menu-models.json'
        if not path.exists():
            cache = Path.home() / '.codex/models_cache.json'
            models = json.loads(cache.read_text()).get('models', []) if cache.exists() else []
            if self.settings.enable_deepseek:
                models += read_models(self.settings.deepseek_catalog)
            self.settings.prepare_state()
            temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
            temporary.write_text(json.dumps({'models': models}) + '\n')
            temporary.chmod(0o600)
            temporary.replace(path)
        return str(path)

    def strip_ui_hint(self, params):
        """Do not pass the desktop-only catalog override into native sessions."""
        config = params.get('config')
        if config and config.get('model_catalog_json') == str(self.settings.state / 'menu-models.json'):
            config.pop('model_catalog_json')

    def engine(self, model):
        """Classify enabled models and reject disabled external-provider names."""
        if model in self.engines:
            return self.engines[model]
        if model and (model.startswith("deepseek-") or model.startswith("claude-code/")):
            raise ValueError("This external model is not enabled in the bridge catalogue")
        return "openai"

    def append(self, response):
        """Append extras only on the final native page, avoiding duplicates."""
        result = dict(response)
        if result.get("nextCursor") is None:
            seen = {item["model"] for item in result["data"]}
            result["data"] = result["data"] + [m for m in self.extra if m["model"] not in seen]
        return result

    def effort(self, model, requested):
        """Normalize stale menu effort values to supported external-model values."""
        entry = next((m for m in self.extra if m["model"] == model), None)
        if not entry:
            return requested
        allowed = {x["reasoningEffort"] for x in entry["supportedReasoningEfforts"]}
        if requested in allowed:
            return requested
        return entry["defaultReasoningEffort"]
