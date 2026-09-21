"""Per-task routing, keeping subscription and external API providers independent."""

import asyncio
import copy
import json
import os
import time
from .catalog import Catalog, PROVIDERS
from .rpc import RpcError
from .leases import Lease, is_owned


SAFE_OPTIONS = (
    "cwd", "approvalPolicy", "approvalsReviewer", "sandbox", "permissions",
    "serviceTier", "personality", "historyMode",
)


def selected_model(params):
    """Honor collaboration mode, which takes precedence in Codex's protocol."""
    collaboration = params.get("collaborationMode") or {}
    return (collaboration.get("settings") or {}).get("model") or params.get("model")


class Router:
    """Route each task under an independent lock, before any model inference."""

    def __init__(self, settings, registry, emit):
        self.settings = settings
        self.registry = registry
        self.emit = emit
        self.catalog = Catalog(settings)
        self.core = None
        self.claude = None
        self.locks = {}
        self.switching = set()
        self.start_options = {}
        self.menu_diagnostics = {}
        self.agents = None
        self.skills = None
        if settings.enable_agents and (settings.enable_deepseek or settings.enable_claude):
            from .agents import AgentManager
            self.agents = AgentManager(self)
        if settings.enable_skills and (settings.enable_deepseek or settings.enable_claude):
            from .skills import SkillManager
            self.skills = SkillManager(self)

    async def event(self, message):
        """Forward native events, suppressing only internal provider-switch closures."""
        params = message.get("params") or {}
        if self.skills and message.get('method') == 'skills/changed':
            self.skills.cache.clear()
        if self.skills and self.skills.intercept(message):
            return
        if self.agents and self.agents.intercept(message):
            return
        if not self.agents and message.get('method') == 'item/tool/call' and params.get('namespace') == 'bridge_agents':
            await self.core.send({'id': message['id'], 'result': {'success': False, 'contentItems': [
                {'type': 'inputText', 'text': 'Bridge delegation is disabled in this adapter. No child was started.'}]}})
            return
        if not self.skills and message.get('method') == 'item/tool/call' and params.get('namespace') == 'bridge_skills':
            await self.core.send({'id':message['id'],'result':{'success':False,'contentItems':[
                {'type':'inputText','text':'Codex skill bridge is disabled; no operation was performed.'}]}})
            return
        if params.get("threadId") in self.switching and message.get("method") in (
            "thread/closed", "thread/status/changed",
        ):
            return
        if self.claude:
            thread_id = params.get("threadId")
            if thread_id in self.claude.active and message.get("method") in ("thread/closed", "thread/status/changed"):
                return
            if isinstance(params.get("thread"), dict):
                self.claude.decorate(params["thread"], include_turns=True)
        await self.external_event(message)

    async def external_event(self, message):
        if self.agents:
            self.agents.observe(message)
        await self.emit(message)

    async def request(self, method, params, *, delegated=False):
        """Handle selective extension points; pass all other methods to native Codex."""
        params = copy.deepcopy(params or {})
        if not self.settings.enable_deepseek and not self.settings.enable_claude:
            return await self.core.call(method, params)
        self.catalog.strip_ui_hint(params)
        if method == "model/list":
            response = self.catalog.append(await self.core.call(method, params))
            self.record_menu("models", [m["model"] for m in response["data"]])
            return response
        if method == "config/read":
            response = await self.core.call(method, params)
            preference = self.registry.preference("menu-model")
            if preference and preference["model"] in self.catalog.engines:
                response["config"].update(preference)
            response["config"]["model_catalog_json"] = self.catalog.ui_catalog_path()
            self.record_menu("catalog_hint", True)
            return response
        if method == "thread/list" and self.claude:
            return await self.claude.list_threads(params)
        if method in ("config/batchWrite", "config/value/write"):
            return await self._write_config(method, params)
        if method == "thread/start":
            return await self._start(params)
        thread_id = params.get("threadId")
        route = self.registry.get(thread_id) if thread_id else None
        if thread_id and not delegated and method in ('turn/start', 'thread/settings/update') and self.registry.agent(thread_id):
            raise ValueError('This is a managed child task. Continue it with bridge_agents.send_message from its parent.')
        if method == "turn/interrupt":
            if self.agents:
                await self.agents.cancel_children(thread_id, params.get('turnId'))
            if route and route["engine"] == "claude":
                return await self.claude.interrupt(thread_id, params.get("turnId"))
            return await self.core.call(method, params)
        if thread_id and method in ("thread/resume", "turn/start", "thread/settings/update"):
            lock = self.locks.setdefault(thread_id, asyncio.Lock())
            async with lock:
                route = self.registry.get(thread_id)
                if method == "thread/resume":
                    return await self._resume(params, route)
                if route and route['engine'] == 'claude':
                    # Serialize the settings snapshot and session acquisition across
                    # adapters, before either can overwrite a running task's route.
                    with Lease(self.settings.state, 'claude-route:' + thread_id):
                        if is_owned(self.settings.state, 'claude:' + thread_id):
                            raise ValueError('This Claude task is active; wait or stop it before starting another turn')
                        return await self._turn_or_settings(method, params, route)
                return await self._turn_or_settings(method, params, route)
        if route and route["engine"] == "claude":
            return await self.claude.request(method, params, route)
        return await self.core.call(method, params)

    def record_menu(self, key, value):
        """Record only menu metadata, never prompts, credentials or file contents."""
        self.menu_diagnostics.update({key: value, "adapter_pid": os.getpid(), "updated_at": int(time.time())})
        self.settings.prepare_state()
        path = self.settings.state / "desktop-menu.json"
        temporary = path.with_name(f"desktop-menu.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(self.menu_diagnostics, indent=2) + "\n")
        temporary.chmod(0o600)
        temporary.replace(path)

    async def close(self):
        """Stop all externally managed Claude sessions before closing native Codex."""
        if self.skills:
            await self.skills.close()
        if self.agents:
            await self.agents.close()
        if self.claude:
            await self.claude.close()

    def _configure(self, params, engine, model):
        result = copy.deepcopy(params)
        result["model"] = model
        result["modelProvider"] = PROVIDERS[engine]
        result["allowProviderModelFallback"] = False
        config = result.setdefault("config", None) or {}
        self.catalog.strip_ui_hint(result)
        config.update(self.settings.provider_config(engine))
        if engine != "openai":
            config["model_reasoning_effort"] = self.catalog.effort(model, config.get("model_reasoning_effort"))
            result["serviceTier"] = None
        result["config"] = config
        return result

    def _remember(self, response, options, locked=False):
        thread = response["thread"]
        provider = response["modelProvider"]
        engine = next((k for k, v in PROVIDERS.items() if v == provider), None)
        # Other existing custom providers remain native and are not claimed by us.
        if engine is None:
            return None
        previous = self.registry.get(thread["id"])
        route = {
            "thread_id": thread["id"], "engine": engine, "provider": provider,
            "model": response["model"], "cwd": response["cwd"],
            "locked": locked or bool(previous and previous["locked"]),
            "session_id": previous["session_id"] if previous else None,
            "options": {
                **(previous["options"] if previous else {}),
                **{key: options[key] for key in SAFE_OPTIONS if key in options},
            },
        }
        # Use effective settings returned by core when the desktop omitted them.
        for key in ('approvalPolicy', 'approvalsReviewer', 'runtimeWorkspaceRoots'):
            if response.get(key) is not None:
                route['options'][key] = response[key]
        if isinstance(response.get('sandbox'), dict):
            route['options']['sandboxPolicy'] = response['sandbox']
        if response.get('reasoningEffort') is not None:
            route['options']['effort'] = response['reasoningEffort']
        self.registry.save(route)
        if engine == "claude":
            self.registry.save_thread(thread)
        return route

    async def _start(self, params):
        capabilities = {k:copy.deepcopy(params[k]) for k in ('dynamicTools','selectedCapabilityRoots') if k in params}
        if self.skills:
            params = self.skills.inject_tools(params)
        if self.agents:
            params = self.agents.inject(params)
        original = copy.deepcopy(params)
        model = selected_model(params)
        engine = self.catalog.engine(model)
        if engine != "openai":
            params = self._configure(params, engine, model)
            params = await self._skill_context(params, params.get('cwd') or os.getcwd())
        response = await self.core.call("thread/start", params)
        self._remember(response, original)
        self.start_options[response["thread"]["id"]] = original
        self.registry.capabilities(response['thread']['id'], capabilities)
        self.registry.task_instructions(response['thread']['id'], original.get('developerInstructions') or '')
        return response

    async def _skill_context(self, params, cwd):
        if self.skills:
            block = await self.skills.context(cwd)
            # Our prior catalogue can be replaced on resume, without replacing user instructions.
            marker = '\n\n<bridge_codex_skills>\n'
            existing = (params.get('developerInstructions') or '').split(marker, 1)[0]
            params['developerInstructions'] = existing + marker + block + '\n</bridge_codex_skills>'
        return params

    async def _resume(self, params, route):
        instructions = params.get('developerInstructions')
        if instructions is None:
            instructions = self.registry.task_instructions(params['threadId'])
        if instructions is not None:
            params['developerInstructions'] = instructions
        if route:
            # Native resume can default cwd to the app-server's directory. Keep
            # this task's workspace and permissions when the UI omits them.
            params["cwd"] = params.get("cwd") or route["cwd"]
            for key in SAFE_OPTIONS:
                if key != "historyMode" and params.get(key) is None and route["options"].get(key) is not None:
                    params[key] = route["options"][key]
            requested = selected_model(params)
            if requested and self.catalog.engine(requested) != route["engine"]:
                # Desktop may supply its global last-used model during restore.
                requested = None
            params = self._configure(params, route["engine"], requested or route["model"])
            if route['engine'] != 'openai' and instructions is not None:
                params = await self._skill_context(params, params['cwd'])
        include_claude_turns = not params.get("excludeTurns", False)
        if route and route["engine"] == "claude":
            params["excludeTurns"] = True
        response = await self.core.call("thread/resume", params)
        # Previously unseen saved tasks are conservatively considered nonempty.
        route = self._remember(response, params, locked=route is None)
        if instructions is not None:
            self.registry.task_instructions(response['thread']['id'], instructions)
        if route and route["engine"] == "claude":
            self.claude.decorate(response["thread"], include_turns=include_claude_turns)
            if self.registry.turns(response["thread"]["id"]):
                # Native core has no concept of a Claude turn, so a resumed
                # thread it never itself produced always reports null
                # pagination cursors here. The desktop's hydration code reads
                # that as "no history" and never follows up with
                # thread/turns/list or thread/items/list, which is what left
                # every restored Claude task looking empty after a restart.
                # Our own dispatcher already serves both methods from the
                # registry, with cursor "0" meaning "start of the sorted
                # page", so handing that back here is enough to make the
                # desktop actually ask for them.
                response["turnsBackwardsCursor"] = "0"
                response["itemsBackwardsCursor"] = "0"
        return response

    async def _switch_empty(self, params, route, engine, model):
        thread_id = route["thread_id"]
        original = self.start_options.get(thread_id, route["options"])
        if original.get("ephemeral"):
            raise ValueError("Choose the provider at thread/start for ephemeral tasks")
        resume = {key: value for key, value in original.items() if key not in (
            "ephemeral", "experimentalRawEvents", "dynamicTools", "selectedCapabilityRoots", "threadSource", "serviceName",
        )}
        if resume.get('developerInstructions') is None:
            instructions = self.registry.task_instructions(thread_id)
            if instructions is not None: resume['developerInstructions'] = instructions
        resume.update({"threadId": thread_id, "excludeTurns": True})
        resume = self._configure(resume, engine, model)
        if engine != 'openai' and resume.get('developerInstructions') is not None:
            resume = await self._skill_context(resume, route['cwd'])
        self.switching.add(thread_id)
        try:
            # A metadata read materializes a persistent but still-empty thread
            # before the native unsubscribe tears down its in-memory session.
            try:
                await self.core.call("thread/read", {"threadId": thread_id, "includeTurns": True})
            except RpcError as exc:
                if exc.error.get("code") != -32601 or str(exc) != "list_turns is not supported yet":
                    raise
            await self.core.call("thread/unsubscribe", {"threadId": thread_id})
            # Native thread/start may acknowledge before its initial rollout is
            # durable. Only retry this metadata operation, never a turn dispatch.
            for attempt in range(11):
                try:
                    response = await self.core.call("thread/resume", resume)
                    break
                except RpcError as exc:
                    if attempt == 10 or "no rollout found for thread id" not in str(exc):
                        raise
                    await asyncio.sleep(0.1)
            return self._remember(response, original)
        finally:
            self.switching.discard(thread_id)

    async def _turn_or_settings(self, method, params, route):
        model = selected_model(params) or (route and route["model"])
        if not model:
            return await self.core.call(method, params)
        engine = self.catalog.engine(model)
        if route is None:
            if engine != "openai":
                raise ValueError("Resume this task before selecting an external provider")
            return await self.core.call(method, params)
        if route["engine"] != engine:
            if route["locked"]:
                raise ValueError("The engine is fixed for this task. Create a new task to use another provider.")
            if engine == "deepseek":
                self.settings.check_deepseek_key()
            route = await self._switch_empty(params, route, engine, model)
        if engine == "deepseek" and method == "turn/start":
            self.settings.check_deepseek_key()
        if engine != "openai":
            params["model"] = model
            params["serviceTier"] = None
            effort = self.catalog.effort(model, params.get("effort"))
            params["effort"] = effort
            collaboration = params.get("collaborationMode")
            if collaboration:
                collaboration["settings"]["reasoning_effort"] = self.catalog.effort(
                    model, collaboration["settings"].get("reasoning_effort"),
                )
        route["model"] = model
        for key in (*SAFE_OPTIONS, 'sandboxPolicy', 'runtimeWorkspaceRoots', 'effort'):
            if params.get(key) is not None:
                route['options'][key] = copy.deepcopy(params[key])
        collaboration = (params.get('collaborationMode') or {}).get('settings') or {}
        if collaboration.get('reasoning_effort') is not None:
            route['options']['effort'] = collaboration['reasoning_effort']
        if params.get('cwd'):
            route['cwd'] = params['cwd']
        if method == "turn/start":
            route["locked"] = True
        # Lock before send: a transport failure may have followed successful dispatch.
        self.registry.save(route)
        if engine == "claude":
            if self.skills and method == 'turn/start':
                params = await self.skills.claude_input(params, route)
            return await self.claude.request(method, params, route)
        return await self.core.call(method, params)

    async def _write_config(self, method, params):
        edits = params.get("edits", [params] if method == "config/value/write" else [])
        model_edit = next((e for e in edits if e.get("keyPath") == "model"), None)
        saved = self.registry.preference("menu-model") or {}
        if model_edit is None:
            if saved.get("model") in self.catalog.engines and any(e.get("keyPath") == "model_reasoning_effort" for e in edits):
                chosen = saved["model"]
            else:
                return await self.core.call(method, params)
        else:
            chosen = model_edit.get("value")
        engine = self.catalog.engine(chosen)
        if engine == "openai":
            response = await self.core.call(method, params)
            self.registry.preference("menu-model", {})
            return response
        # Desktop persists menu choice globally. Keep that preference in our DB;
        # native config must retain its real OpenAI model for normal startup/rollback.
        if params.get("filePath") is not None:
            raise ValueError("External menu models are supported only for the default user configuration")
        original = await self.core.call("config/read", {"includeLayers": True})
        layer = next((entry for entry in original.get("layers", []) if
                      entry["name"].get("type") == "user" and not entry["name"].get("profile")), None)
        if layer is None:
            raise ValueError("Cannot locate the native user configuration metadata")
        expected = params.get("expectedVersion")
        if expected is not None and expected != layer["version"]:
            raise ValueError("Configuration changed; refresh settings before selecting a model")
        preference = {**(saved if saved.get("model") == chosen else {}), "model": chosen}
        remaining = []
        for edit in edits:
            if edit.get("keyPath") in ("model", "model_reasoning_effort"):
                key = edit["keyPath"]
                preference[key] = edit.get("value")
            else:
                remaining.append(edit)
        if remaining:
            params["edits"] = remaining
            response = await self.core.call("config/batchWrite", params)
        else:
            # A virtual menu preference changes no native file. Return its current
            # version so subsequent real config writes retain concurrency checks.
            response = {"status": "ok", "filePath": layer["name"]["file"], "version": layer["version"]}
        self.registry.preference("menu-model", preference)
        return response
