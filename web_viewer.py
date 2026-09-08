#!/usr/bin/env python3
"""Local browser dashboard for the Hermes model-router JSONL log."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    import yaml
except ImportError:
    yaml = None

from agent_activity import load_agent_activity
from view_log import DEFAULT_LOG, load_entries

DEFAULT_STATE_DB = Path("~/.hermes/state.db").expanduser()
DEFAULT_AGENT_LOG = Path("~/.hermes/logs/agent.log").expanduser()
DEFAULT_BRIDGE_LIFECYCLE = Path("~/.hermes/logs/claude-code-bridge.jsonl").expanduser()
DEFAULT_ROOT_LIMIT = 10
RAW_HISTORY_LIMIT = 10000
CONFIG_PATH = Path("~/.hermes/plugins/model_router/router_config.yaml").expanduser()


def _router_status() -> dict:
    """Cooldown state and per-account load, read through the router's own code.

    Imported lazily and defensively: the dashboard is a standalone script and
    must keep serving the log even if the router package cannot be imported.
    Recomputing either of these here instead would let the panel and the routing
    decision disagree about which tiers are available.
    """
    empty = {"cooldowns": {}, "load": {}, "window_minutes": 0, "routable": []}
    try:
        import sys

        parent = str(Path(__file__).resolve().parent.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from model_router import (
            _load_config,
            _read_cooldown_state,
            _recent_account_load,
            _tier_cooldown_remaining,
        )
    except Exception:
        return empty
    try:
        config = _load_config()
        window = int((config.get("usage_report") or {}).get("window_seconds") or 3600)
        recorded = _read_cooldown_state(config).get("tiers") or {}
        cooldowns = {}
        for tier in ("luna", "spark", "terra", "sol", "opus5", "qwen"):
            remaining = _tier_cooldown_remaining(tier, config)
            if remaining > 0:
                cooldowns[tier] = {
                    "seconds": int(remaining),
                    "reason": str((recorded.get(tier) or {}).get("reason") or ""),
                }
        return {
            "cooldowns": cooldowns,
            "load": _recent_account_load(config, window),
            "window_minutes": max(1, window // 60),
            # Which tiers can actually hold the orchestrator role. A tier the
            # router has no model entry for cannot: picking it raises a KeyError
            # on the first routing decision. Served rather than hardcoded so the
            # dropdown cannot drift from the router's own tier map again.
            "routable": sorted(config.get("models") or {}),
        }
    except Exception:
        return empty


HTML = r'''<!doctype html>
<html lang="hu">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Home Lab</title>
<style>
:root{color-scheme:dark;--bg:#080b12;--panel:#111723;--border:#253044;--text:#e9eef8;--muted:#8997ad;--luna:#61dafb;--spark:#f3f6fb;--terra:#88e36f;--sol:#ffb454;--opus5:#d695ff;--sonnet5:#9fb4ff;--qwen:#38d9a9;--accent:#a78bfa}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#17203a 0,transparent 35%),var(--bg);color:var(--text);font:14px/1.5 Inter,Segoe UI,system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:28px}h1{font-size:26px;margin:0}.sub{color:var(--muted);margin:4px 0 22px}.toolbar,.cards{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}.toolbar{align-items:end;background:rgba(17,23,35,.86);padding:14px;border:1px solid var(--border);border-radius:14px;backdrop-filter:blur(10px)}
label{display:grid;gap:5px;color:var(--muted);font-size:12px}input,select,button{background:#0b111c;color:var(--text);border:1px solid var(--border);border-radius:8px;padding:9px 11px;font:inherit}input[type=search]{min-width:260px}button{cursor:pointer}button:hover{border-color:var(--accent)}button:disabled{cursor:wait;opacity:.72}.status.refreshing{color:#c4b5fd}.check{display:flex;align-items:center;gap:7px;padding:9px 2px}.check input{accent-color:var(--accent)}
.card{min-width:135px;flex:1;background:linear-gradient(145deg,#151d2c,#0e1420);border:1px solid var(--border);border-radius:14px;padding:15px}.card .n{font-size:25px;font-weight:750}.card .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}.luna .n{color:var(--luna)}.spark .n{color:var(--spark)}.terra .n{color:var(--terra)}.sol .n{color:var(--sol)}.opus5 .n{color:var(--opus5)}.sonnet5 .n{color:var(--sonnet5)}.qwen .n{color:var(--qwen)}
.table-wrap{overflow:auto;border:1px solid var(--border);border-radius:14px;background:rgba(13,18,29,.92)}table{border-collapse:collapse;table-layout:fixed;width:1405px;min-width:100%}th{position:sticky;top:0;background:#161e2c;color:var(--muted);font-size:11px;letter-spacing:.07em;text-align:left;text-transform:uppercase;user-select:none}th,td{padding:11px 13px;border-bottom:1px solid #1e2838;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.prompt-text{min-width:0}.word-wrap .prompt-text{white-space:normal;overflow:visible;text-overflow:clip;overflow-wrap:anywhere}.resizer{position:absolute;z-index:2;top:0;right:-4px;width:9px;height:100%;cursor:col-resize;touch-action:none}.resizer::after{content:'';position:absolute;left:4px;top:20%;width:1px;height:60%;background:#3b4a62}.resizer:hover::after,.resizer.dragging::after{width:2px;background:var(--accent)}body.resizing{cursor:col-resize;user-select:none}tbody tr:hover{background:#151d2b}code{color:#b9c5d8}.pill{display:inline-block;border:1px solid currentColor;border-radius:99px;padding:2px 8px;font-size:12px;font-weight:700;margin-right:4px}.pill.luna{color:var(--luna)}.pill.spark{color:var(--spark)}.pill.terra{color:var(--terra)}.pill.sol{color:var(--sol)}.pill.opus5{color:var(--opus5)}.pill.sonnet5{color:var(--sonnet5)}.pill.qwen{color:var(--qwen)}.route{white-space:nowrap}.reason{color:#c2ccdb}.running-agent-pill{display:inline-block;margin-left:8px;padding:2px 7px;border:1px solid #88e36f;border-radius:99px;color:#88e36f;font-size:10px;font-weight:800;letter-spacing:.06em;vertical-align:middle;animation:agentPulse 1.4s ease-in-out infinite}@keyframes agentPulse{50%{box-shadow:0 0 11px rgba(136,227,111,.7)}}.play-indicator{display:inline-flex;align-items:center;justify-content:center;width:19px;height:19px;margin-left:8px;border-radius:50%;background:#54d66a;color:#07110b;font-size:10px;font-weight:900;vertical-align:middle;box-shadow:0 0 12px rgba(84,214,106,.75);animation:agentPulse 1.4s ease-in-out infinite}.active-router-row{background:rgba(84,214,106,.055)}.calls,.expand{text-align:center}.prompt-toggle{border:0;background:transparent;padding:0;color:var(--accent);font-size:15px;line-height:1}.prompt-toggle:hover{border:0;color:#c4b5fd}.detail>td{padding:10px 13px 14px;background:#0b111c;overflow:visible}.details-table{width:calc(100% - 28px);min-width:0;margin-left:28px;border:1px solid #253044;border-radius:8px;table-layout:fixed}.details-table th{position:static;background:#111927}.details-table th,.details-table td{padding:8px 10px;font-size:12px}.details-table tbody tr:last-child td{border-bottom:0}.status{margin-left:auto;color:var(--muted);padding:9px 4px}.empty{text-align:center;color:#8997ad;padding:40px}.error{color:#ff6b7a} @media(max-width:700px){main{padding:16px}input[type=search]{min-width:180px}.status{width:100%;margin:0}}
</style><style>.agents{margin-top:22px;padding:18px;border:1px solid var(--border);border-radius:14px;background:linear-gradient(145deg,#151d2c,#0e1420)}.agents h2{margin:0;font-size:18px}.agent-parent{margin-top:12px;border-top:1px solid #253044;padding-top:12px}.agent-session{color:#c4b5fd;font-size:12px;letter-spacing:.06em}.agent-child{display:grid;grid-template-columns:10px 1fr auto;gap:10px;align-items:center;margin-top:9px;padding:10px 12px;border-radius:10px;background:#0b111c}.agent-dot{width:9px;height:9px;border-radius:50%;background:#8997ad}.agent-dot.running{background:#88e36f;box-shadow:0 0 12px #88e36f}.agent-goal{font-weight:700}.agent-activity{color:#a78bfa;font-size:12px;margin-top:2px}.agent-meta{color:var(--muted);font-size:12px;text-align:right}.agent-empty{color:var(--muted);padding:12px 0}</style><style>.lab-header{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:18px}.lab-kicker{color:#a78bfa;font-size:11px;font-weight:800;letter-spacing:.14em}.lab-title{font-size:30px;font-weight:800;letter-spacing:-.04em}.tabs{display:flex;gap:6px;padding:5px;border:1px solid var(--border);border-radius:12px;background:#0b111c}.tab{border:0;background:transparent;color:var(--muted);font-weight:700}.tab.active{background:#252039;color:#e9ddff}.panel[hidden]{display:none}.panel-heading{font-size:18px;font-weight:750;margin:0 0 4px}</style><style>.console{margin:10px 0 4px 19px;border:1px solid #2c3951;border-radius:10px;background:#080d16}.console summary,.agent-history summary{cursor:pointer;padding:9px 11px;color:#c4b5fd;font-weight:700}.console-event{border-top:1px solid #1e2838}.console-event.compact{padding:6px 11px;color:#c6d0df;font:12px ui-monospace,SFMono-Regular,Consolas,monospace}.console-label{padding:7px 11px;color:#88e36f;font:12px ui-monospace,SFMono-Regular,Consolas,monospace}.console pre{margin:0;padding:0 11px 11px;max-height:220px;overflow:auto;white-space:pre-wrap;word-break:break-word;color:#c6d0df;font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}.console-empty{padding:11px;color:var(--muted)}.agent-history{margin-top:20px;border-top:1px solid #253044}.history-item{padding:7px 12px;color:var(--muted);border-top:1px solid #1e2838}</style><style>.settings{margin-top:22px;display:flex;flex-direction:column;gap:18px}.settings-section{padding:18px;border:1px solid var(--border);border-radius:14px;background:linear-gradient(145deg,#151d2c,#0e1420)}.settings-section h3{margin:0 0 14px;font-size:16px;color:var(--accent);text-transform:uppercase;letter-spacing:.1em}.toggle-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}.toggle-item{display:flex;align-items:center;justify-content:space-between;padding:14px;border:1px solid var(--border);border-radius:10px;background:#0b111c}.toggle-item.disabled{opacity:.5;border-color:#1a2033}.toggle-label{display:flex;flex-direction:column;gap:2px}.toggle-name{font-weight:700;font-size:14px}.toggle-desc{font-size:11px;color:var(--muted)}.switch{position:relative;width:44px;height:24px}.switch input{opacity:0;width:0;height:0}.switch .slider{position:absolute;cursor:pointer;top:0;left:0;right:0;bottom:0;background:#253044;border-radius:24px;transition:.2s}.switch .slider::before{position:absolute;content:'';height:18px;width:18px;left:3px;bottom:3px;background:#8997ad;border-radius:50%;transition:.2s}.switch input:checked+.slider{background:#88e36f}.switch input:checked+.slider::before{transform:translateX(20px);background:#07110b}.default-model-row{display:flex;align-items:end;gap:12px;padding:14px;border:1px solid var(--border);border-radius:10px;background:#0b111c}.default-model-row label{flex:0 0 auto}.default-model-row select{min-width:200px}.save-settings{align-self:flex-end;padding:10px 20px;background:var(--accent);color:#fff;border:0;border-radius:8px;font-weight:700;cursor:pointer}.save-settings:hover{background:#8b72f0}.save-settings:disabled{opacity:.6;cursor:wait}.settings-status{margin-left:auto;font-size:12px;color:var(--muted)}.cooldown-pill{display:inline-block;margin-top:4px;padding:2px 7px;border-radius:999px;background:#3a2418;border:1px solid #7c4a25;color:#ffbe8a;font-size:10px;font-weight:700;letter-spacing:.04em;white-space:nowrap}.toggle-item.cooling{border-color:#7c4a25}.account-load{margin-top:14px;padding:12px 14px;border:1px solid var(--border);border-radius:10px;background:#0b111c;font-size:12px;color:var(--muted)}.account-load b{color:#e6edf6;font-weight:700}.account-load .idle{color:#88e36f}</style><style>.command-frame{display:block;width:100%;height:calc(100vh - 180px);min-height:680px;border:1px solid var(--border);border-radius:14px;background:#111723}</style>
</head>
<body><main>
<div class="lab-header"><div><div class="lab-kicker" data-i18n="lab.kicker">HERMES · LOCAL OBSERVABILITY</div><div class="lab-title" data-i18n="lab.title">AI Home Lab</div><div class="sub" data-i18n="lab.sub">Modellek, háttéragentek és élő munkafolyamatok egy helyen</div></div><nav class="tabs" aria-label="AI Home Lab nézetek"><button class="tab active" data-tab="router"><span data-i18n="tab.router">Model Router</span></button><button class="tab" data-tab="settings"><span data-i18n="tab.settings">Beállítások</span></button><button class="tab" data-tab="command"><span data-i18n="tab.command">Hermes Command Center</span></button></nav></div>
<section id="router-panel" class="panel" hidden><h2 class="panel-heading"><span data-i18n="router.heading">Model Router</span></h2><div class="sub" data-i18n="router.sub">Élő JSONL routing napló · automatikus frissítés 3 másodpercenként</div>
<div class="toolbar">
<label><span data-i18n="router.tier.label">Modell</span><select id="tier"><option value="" data-i18n="router.tier.all">Mind</option><option>luna</option><option>spark</option><option>terra</option><option>sol</option><option>opus5</option><option>qwen</option></select></label>
<label>Keresés<input id="search" type="search" data-i18n-placeholder="router.search.placeholder"></label>
<label><span data-i18n="router.last.label">Utolsó root promptok</span><select id="last"><option>1</option><option>5</option><option selected>10</option></select></label>
<label class="check"><input id="grouped" type="checkbox" checked> <span data-i18n="router.grouped">Promptonként összevonva</span></label>
<label class="check"><input id="word-wrap" type="checkbox"> <span data-i18n="router.wordwrap">Sortörés</span></label>
<label class="check"><input id="auto" type="checkbox" checked> <span data-i18n="router.auto">Automatikus frissítés</span></label>
<button id="refresh" data-i18n="router.refresh">Frissítés</button><span id="status" class="status" data-i18n="router.loading">Betöltés…</span>
</div>
<div class="cards"><div class="card"><div class="n" id="total">0</div><div class="k" data-i18n="card.total">Összes routing döntés</div></div><div class="card luna"><div class="n" id="luna">0</div><div class="k" data-i18n="card.luna">GPT-5.6 Luna</div></div><div class="card spark"><div class="n" id="spark">0</div><div class="k" data-i18n="card.spark">GPT-5.3 Spark</div></div><div class="card terra"><div class="n" id="terra">0</div><div class="k" data-i18n="card.terra">GPT-5.6 Terra</div></div><div class="card sol"><div class="n" id="sol">0</div><div class="k" data-i18n="card.sol">GPT-5.6 Sol</div></div><div class="card opus5"><div class="n" id="opus5">0</div><div class="k" data-i18n="card.opus5">Claude Opus 5</div></div><div class="card sonnet5"><div class="n" id="sonnet5">0</div><div class="k" data-i18n="card.sonnet5">Claude Sonnet 5</div></div><div class="card qwen"><div class="n" id="qwen">0</div><div class="k" data-i18n="card.qwen">Qwen 3.7 Plus</div></div></div>
<div id="runs" class="router-runs" aria-live="polite"></div><div class="table-wrap" hidden><table id="log-table"><colgroup><col style="width:55px"><col style="width:110px"><col style="width:90px"><col style="width:350px"><col style="width:90px"><col style="width:230px"><col style="width:480px"></colgroup><thead><tr><th class="expand"></th><th data-i18n="th.date">Dátum</th><th data-i18n="th.time">Idő (CET/CEST)</th><th data-i18n="th.prompt">Prompt</th><th class="calls" data-i18n="th.calls">Hívások</th><th data-i18n="th.route">Útvonal</th><th data-i18n="th.reason">Indok</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<section id="settings-panel" class="panel" hidden><h2 class="panel-heading"><span data-i18n="settings.heading">Beállítások</span></h2><div class="sub" data-i18n="settings.sub">Modellek hívhatósága és alapértelmezett modell</div><div class="settings"><div class="settings-section"><h3 data-i18n="settings.callable">Modellek hívhatósága</h3><div class="toggle-grid" id="callable-toggles"></div><div class="account-load" id="account-load"></div></div><div class="settings-section"><h3 data-i18n="settings.default.heading">Alapértelmezett modell (Orchestrator)</h3><div class="default-model-row"><label><span data-i18n="settings.default.desc">Ez a modell látja el az alapértelmezett routingot és az orchestrator szerepkört</span><select id="default-model-select"></select></label><span class="settings-status" id="settings-status"></span></div></div><div class="settings-section"><h3 data-i18n="settings.language">Nyelv</h3><div class="default-model-row"><label><span data-i18n="settings.language">Nyelv</span><select id="language-select"><option value="en" data-i18n="settings.lang.en">English</option><option value="hu" data-i18n="settings.lang.hu">Magyar</option></select></label></div></div></div></section>
<section id="command-panel" class="panel" hidden><h2 class="panel-heading"><span data-i18n="tab.command">Hermes Command Center</span></h2><div class="sub">A Hermes hivatalos helyi kezelőfelülete</div><iframe class="command-frame" title="Hermes Command Center" src="http://127.0.0.1:9119/"></iframe></section>
</main>
<script>
const $=id=>document.getElementById(id);
// ── i18n ──────────────────────────────────────────────────────────
const I18N = {
  en: {
    // Header
    'lab.kicker': 'HERMES · LOCAL OBSERVABILITY',
    'lab.title': 'AI Home Lab',
    'lab.sub': 'Models, background agents and live workflows in one place',
    // Tabs
    'tab.router': 'Model Router',
    'tab.settings': 'Settings',
    'tab.command': 'Hermes Command Center',
    // Router panel
    'router.heading': 'Model Router',
    'router.sub': 'Live JSONL routing log · auto-refresh every 3 seconds',
    'router.tier.label': 'Model',
    'router.tier.all': 'All',
    'router.search.placeholder': 'prompt, model or reason…',
    'router.last.label': 'Last root prompts',
    'router.grouped': 'Grouped by prompt',
    'router.wordwrap': 'Word wrap',
    'router.auto': 'Auto refresh',
    'router.refresh': 'Refresh',
    'router.loading': 'Loading…',
    // Cards
    'card.total': 'Total routing decisions',
    'card.luna': 'GPT-5.6 Luna',
    'card.spark': 'GPT-5.3 Spark',
    'card.terra': 'GPT-5.6 Terra',
    'card.sol': 'GPT-5.6 Sol',
    'card.opus5': 'Claude Opus 5',
    'card.sonnet5': 'Claude Sonnet 5',
    'card.qwen': 'Qwen 3.7 Plus',
    // Table headers
    'th.date': 'Date',
    'th.time': 'Time (CET/CEST)',
    'th.prompt': 'Prompt',
    'th.calls': 'Calls',
    'th.route': 'Route',
    'th.reason': 'Reason',
    // Misc router
    'no.entries': 'No matching entries.',
    'not.recoverable': 'Not recoverable',
    'details.close': 'Close details',
    'details.open': 'Open API calls',
    'related.missing': 'Related earlier message not found.',
    'subagent.close': 'Close sub-agents',
    'subagent.open': 'Open sub-agents',
    'subagent.running': 'Sub-agent running',
    'subagent.done': 'Finished',
    'main.agent': 'Main agent',
    'root.continuation': 'INTERNAL ROUTER CONTINUATIONS',
    'root.continuation.title': 'Not a delegated worker: internal continuations and routing decisions of the main thread between calls.',
    'close.router.steps': 'Close internal router steps',
    'open.router.steps': 'Open internal router steps',
    // Agent panel
    'agents.heading': 'Delegated sub-agents',
    'agents.running.done.error': 'running · completed · errors',
    'agents.recent': 'recent runs',
    'agents.none': 'No delegated sub-agents to display.',
    'agents.main.thread': 'MAIN THREAD',
    'agents.prev.task': 'Previous main task',
    'agents.reason.default': 'Independent subtask',
    'agents.calls': 'calls',
    'agents.close.last': 'Last',
    'agents.close.steps': 'tool events',
    'agents.no.events': 'No stored tool events for this agent yet.',
    'agents.recently.done': 'Recently completed',
    'agents.main.task': 'Main task',
    'agents.open.tasks': 'Open subtasks',
    'agents.close.tasks': 'Close subtasks',
    'agents.subtasks': 'subtasks',
    'agents.running': 'Running',
    'agents.done': 'Done',
    'agents.task.desc': 'Task description',
    'agents.no.desc': 'No saved task description.',
    'agents.close.inner': 'Close inner tasks',
    'agents.open.inner': 'Open inner tasks',
    'agents.below.root': 'Below root',
    'agents.requested.ro': 'Requested READ-ONLY',
    'agents.requested.ro.title': 'The main agent explicitly requested this sub-agent for read-only/verification only.',
    // Settings panel
    'settings.heading': 'Settings',
    'settings.sub': 'Model callability and default model',
    'settings.callable': 'Model callability',
    'settings.cooling': 'cooling down',
    'settings.load.title': 'Recent load per account',
    'settings.load.window': 'last {n} min',
    'settings.load.idle': 'no calls — prefer it for an independent leaf',
    'settings.load.counts': 'call counts from the route log, not quota readings',
    'settings.load.empty': 'No routed calls in the window.',
    'settings.default.heading': 'Default model (Orchestrator)',
    'settings.default.desc': 'This model handles default routing and the orchestrator role',
    'settings.language': 'Language',
    'settings.lang.en': 'English',
    'settings.lang.hu': 'Magyar',
    // Model descriptions
    'model.desc.luna': 'Fast, simple tasks',
    'model.desc.spark': 'Read-only delegated work',
    'model.desc.terra': 'General orchestrator',
    'model.desc.sol': 'Security-critical, design',
    'model.desc.opus5': 'Claude account — consequential work',
    'model.desc.sonnet5': 'Claude account — default choice',
    'model.desc.qwen': 'Alternative model',
    // Settings messages
    'settings.saving': 'Saving...',
    'settings.saved': 'Saved ✓',
    'settings.error.unknown': 'Unknown error',
    'settings.error.prefix': 'Error: ',
    'settings.error.load': 'Failed to load settings:',
    // Status
    'status.refreshing': 'Refreshing…',
    'status.loading': 'Loading…',
    'status.error.prefix': 'Error: ',
    'status.error.kept': ' · previous list kept',
    // Execution tree
    'exec.below.root': 'Below root',
    'exec.open.subtasks': 'Open subtasks',
    'exec.close.subtasks': 'Close subtasks',
    'exec.no.desc': 'No saved task description.',
    'exec.opus.review': 'Opus review',
    'exec.internal.step': 'Internal router step',
    'exec.read.only': 'READ-ONLY',
    'exec.requested.ro': 'REQUESTED READ-ONLY',
    // Misc
    'internal.router.step': 'Internal router step',
    'resizer.title': 'Drag to resize column · double-click: reset',
    'play.running.agent': 'Currently running main agent',
    'play.running.sub': 'AGENTS RUNNING NOW',
    'status.refresh.short': 'Refresh…',
    'status.kept': ' · previous list shown',
    'status.rootprompts': 'root prompts',
    'status.locale': 'en-GB',
    'total.routing.decisions': 'TOTAL ROUTING DECISIONS',
    'run.worker.routing': 'worker-routing',
    'agents.sum.running': 'running',
    'agents.sum.completed': 'completed',
    'agents.sum.failed': 'errors',
    'agents.main.agent': 'MAIN AGENT',
    'agents.none.active': 'No active background agents.',
    'agents.pill.running': 'RUNNING NOW',
    'agents.pill.agent': 'AGENT',
    'state.running.short': 'RUNNING',
    'state.done.short': 'DONE',
    'state.success.short': 'SUCCESS',
    'state.error.short': 'ERROR',
    'duration.h': 'h',
    'duration.m': 'm',
    'duration.s': 's',
    'exec.no.route': 'no stored route data',
    'exec.external': 'EXTERNAL',
    'exec.own': 'own',
    'exec.total': 'total',
    'root.continuation.reason': 'System continuation · routing decisions between calls',
    'settings.error.save': 'Failed to save settings:',
    'agents.unknown.model': 'unknown model',
    'play.running.subagent': 'Currently running sub-agent',
    'label.model.short': 'model',
    'label.active.turn': 'Active turn: ',
    'exec.tree.label': 'Execution tree',
    'th.time.short': 'Time',
  },
  hu: {
    'lab.kicker': 'HERMES · LOCAL OBSERVABILITY',
    'lab.title': 'AI Home Lab',
    'lab.sub': 'Modellek, háttéragentek és élő munkafolyamatok egy helyen',
    'tab.router': 'Model Router',
    'tab.settings': 'Beállítások',
    'tab.command': 'Hermes Command Center',
    'router.heading': 'Model Router',
    'router.sub': 'Élő JSONL routing napló · automatikus frissítés 3 másodpercenként',
    'router.tier.label': 'Modell',
    'router.tier.all': 'Mind',
    'router.search.placeholder': 'prompt, modell vagy indok…',
    'router.last.label': 'Utolsó root promptok',
    'router.grouped': 'Promptonként összevonva',
    'router.wordwrap': 'Sortörés',
    'router.auto': 'Automatikus frissítés',
    'router.refresh': 'Frissítés',
    'router.loading': 'Betöltés…',
    'card.total': 'Összes routing döntés',
    'card.luna': 'GPT-5.6 Luna',
    'card.spark': 'GPT-5.3 Spark',
    'card.terra': 'GPT-5.6 Terra',
    'card.sol': 'GPT-5.6 Sol',
    'card.opus5': 'Claude Opus 5',
    'card.sonnet5': 'Claude Sonnet 5',
    'card.qwen': 'Qwen 3.7 Plus',
    'th.date': 'Dátum',
    'th.time': 'Idő (CET/CEST)',
    'th.prompt': 'Prompt',
    'th.calls': 'Hívások',
    'th.route': 'Útvonal',
    'th.reason': 'Indok',
    'no.entries': 'Nincs a szűrésnek megfelelő bejegyzés.',
    'not.recoverable': 'Nem visszakereshető',
    'details.close': 'Részletek bezárása',
    'details.open': 'API-hívások megnyitása',
    'related.missing': 'A kapcsolódó korábbi üzenet nem található.',
    'subagent.close': 'Mellékagentek bezárása',
    'subagent.open': 'Mellékagentek megnyitása',
    'subagent.running': 'Mellékagent fut',
    'subagent.done': 'Kész',
    'main.agent': 'Fő agent',
    'root.continuation': 'BELSŐ ROUTER FOLYTATÁSOK',
    'root.continuation.title': 'Nem delegált worker: a főszál belső, hívások közötti folytatásai és routing-döntései.',
    'close.router.steps': 'Belső router-lépések bezárása',
    'open.router.steps': 'Belső router-lépések megnyitása',
    'agents.heading': 'Delegált mellékszálak',
    'agents.running.done.error': 'fut · kész · hiba',
    'agents.recent': 'legutóbbi futás',
    'agents.none': 'Nincs megjeleníthető delegált mellékszál.',
    'agents.main.thread': 'FŐSZÁL',
    'agents.prev.task': 'Korábbi fő feladat',
    'agents.reason.default': 'Önálló részfeladat',
    'agents.calls': 'hívás',
    'agents.close.last': 'Utolsó',
    'agents.close.steps': 'lépés',
    'agents.no.events': 'Még nincs tárolt tool-esemény ehhez az agenthez.',
    'agents.recently.done': 'Legutóbb befejezett munkák',
    'agents.main.task': 'Fő feladat',
    'agents.open.tasks': 'Alfeladatok megnyitása',
    'agents.close.tasks': 'Alfeladatok bezárása',
    'agents.subtasks': 'mellékszál',
    'agents.running': 'Fut',
    'agents.done': 'Kész',
    'agents.task.desc': 'Feladatleírás',
    'agents.no.desc': 'Nincs megőrzött feladatleírás.',
    'agents.close.inner': 'Belső feladatok bezárása',
    'agents.open.inner': 'Belső feladatok megnyitása',
    'agents.below.root': 'Gyökér alatt',
    'agents.requested.ro': 'KÉRT READ-ONLY',
    'agents.requested.ro.title': 'A fő agent kifejezetten csak olvasási/ellenőrzési feladatra kérte ezt a mellékszálat.',
    'settings.heading': 'Beállítások',
    'settings.sub': 'Modellek hívhatósága és alapértelmezett modell',
    'settings.callable': 'Modellek hívhatósága',
    'settings.cooling': 'hűl',
    'settings.load.title': 'Fogyás accountonként',
    'settings.load.window': 'utolsó {n} perc',
    'settings.load.idle': 'nincs hívás — ide érdemes önálló leafet adni',
    'settings.load.counts': 'hívásszám a route logból, nem kvótaadat',
    'settings.load.empty': 'Nincs routolt hívás az ablakban.',
    'settings.default.heading': 'Alapértelmezett modell (Orchestrator)',
    'settings.default.desc': 'Ez a modell látja el az alapértelmezett routingot és az orchestrator szerepkört',
    'settings.language': 'Nyelv',
    'settings.lang.en': 'English',
    'settings.lang.hu': 'Magyar',
    'model.desc.luna': 'Gyors, egyszerű feladatok',
    'model.desc.spark': 'Read-only delegált munkák',
    'model.desc.terra': 'Általános orchestrator',
    'model.desc.sol': 'Biztonságkritikus, design',
    'model.desc.opus5': 'Claude account — súlyosabb munka',
    'model.desc.sonnet5': 'Claude account — alapértelmezett',
    'model.desc.qwen': 'Alternatív modell',
    'settings.saving': 'Mentés...',
    'settings.saved': 'Mentve ✓',
    'settings.error.unknown': 'Ismeretlen hiba',
    'settings.error.prefix': 'Hiba: ',
    'settings.error.load': 'Nem sikerült betölteni:',
    'status.refreshing': 'Frissítés…',
    'status.loading': 'Betöltés…',
    'status.error.prefix': 'Hiba: ',
    'status.error.kept': ' · a korábbi lista megmaradt',
    'exec.below.root': 'Gyökér alatt',
    'exec.open.subtasks': 'Alfeladatok megnyitása',
    'exec.close.subtasks': 'Alfeladatok bezárása',
    'exec.no.desc': 'Nincs megőrzött feladatleírás.',
    'exec.opus.review': 'Opus review',
    'exec.internal.step': 'Belső router-lépés',
    'exec.read.only': 'READ-ONLY',
    'exec.requested.ro': 'KÉRT READ-ONLY',
    'internal.router.step': 'Belső router-lépés',
    'resizer.title': 'Húzd az oszlop szélességének módosításához · dupla kattintás: alaphelyzet',
    'play.running.agent': 'Éppen futó fő agent',
    'play.running.sub': 'ÉPPEN FUT ·',
    'status.refresh.short': 'Frissítés…',
    'status.kept': ' · a korábbi lista látszik',
    'status.rootprompts': 'root prompt',
    'status.locale': 'hu-HU',
    'total.routing.decisions': 'ÖSSZES ROUTING DÖNTÉS',
    'run.worker.routing': 'worker-routing',
    'agents.sum.running': 'fut',
    'agents.sum.completed': 'kész',
    'agents.sum.failed': 'hiba',
    'agents.main.agent': 'FŐ AGENT',
    'agents.none.active': 'Nincs aktív háttéragent.',
    'agents.pill.running': 'ÉPPEN FUT',
    'agents.pill.agent': 'AGENT',
    'state.running.short': 'FUT',
    'state.done.short': 'KÉSZ',
    'state.success.short': 'SIKER',
    'state.error.short': 'HIBA',
    'duration.h': 'ó',
    'duration.m': 'p',
    'duration.s': 'mp',
    'exec.no.route': 'nincs tárolt route-adat',
    'exec.external': 'KÜLSŐ',
    'exec.own': 'saját',
    'exec.total': 'összesített',
    'root.continuation.reason': 'Rendszerfolytatás · hívások közötti routing-döntések',
    'settings.error.save': 'Nem sikerült menteni:',
    'agents.unknown.model': 'ismeretlen modell',
    'play.running.subagent': 'Éppen futó mellékagent',
    'label.model.short': 'modell',
    'label.active.turn': 'Aktív turn: ',
    'exec.tree.label': 'Végrehajtási fa',
    'th.time.short': 'Idő',
  }
};
let currentLang = localStorage.getItem('model-router-lang') || 'en';
function t(key) { return (I18N[currentLang] && I18N[currentLang][key]) || (I18N.en[key]) || key; }
function applyLanguage() {
  document.documentElement.lang = currentLang === 'hu' ? 'hu' : 'en';
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    const attr = el.getAttribute('data-i18n-attr');
    const val = t(key);
    if (attr) { el.setAttribute(attr, val); }
    else { el.textContent = val; }
  });
  // Update placeholders separately
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    el.placeholder = t(el.getAttribute('data-i18n-placeholder'));
  });
  // Update titles
  document.querySelectorAll('[data-i18n-title]').forEach(el => {
    el.title = t(el.getAttribute('data-i18n-title'));
  });
  // Re-render settings if visible
  if (currentConfig) renderSettings();
  // Re-render dynamic content
  if (typeof render === 'function') { try { render(); } catch(e){} }
  if (typeof renderAgents === 'function') { try { renderAgents(agentActivity); } catch(e){} }
}
let entries=[];let selectedTab=localStorage.getItem('ai-home-lab-tab')||null;const expandedPrompts=new Set(),promptExpansionKey='model-router-expanded-prompts-v1';function persistedPromptExpansions(){try{return new Set(JSON.parse(localStorage.getItem(promptExpansionKey)||'[]'))}catch(e){return new Set()}}function isPromptExpanded(key){return expandedPrompts.has(key)||persistedPromptExpansions().has(key)}function setPromptExpanded(key,open){const persisted=persistedPromptExpansions();open?persisted.add(key):persisted.delete(key);localStorage.setItem(promptExpansionKey,JSON.stringify([...persisted]));open?expandedPrompts.add(key):expandedPrompts.delete(key)}const descriptionExpansionKey='model-router-description-expansion-v1';function descriptionExpansions(){try{return JSON.parse(localStorage.getItem(descriptionExpansionKey)||'{}')}catch(e){return {}}}function createDescriptionDetails(key,text){const node=document.createElement('details'),state=descriptionExpansions();node.className='agent-worker-goal-details';node.open=!!state[key];const summary=document.createElement('summary');summary.textContent=t('agents.task.desc');const body=document.createElement('div');body.className='agent-worker-goal';body.textContent=text||t('agents.no.desc');node.append(summary,body);node.addEventListener('toggle',()=>{const next=descriptionExpansions();next[key]=node.open;localStorage.setItem(descriptionExpansionKey,JSON.stringify(next))});return node}
function setTab(name,remember=true){if(name==='agents')name='router';selectedTab=name;if(remember)localStorage.setItem('ai-home-lab-tab',name);$('router-panel').hidden=name!=='router';$('settings-panel').hidden=name!=='settings';$('command-panel').hidden=name!=='command';document.querySelectorAll('.tab').forEach(tab=>tab.classList.toggle('active',tab.dataset.tab===name));if(name==='settings')loadSettings();}

// Settings management
let currentConfig=null;
async function loadSettings(){
  try{
    const response=await fetch('/api/config',{cache:'no-store'});
    if(!response.ok)throw new Error(`HTTP ${response.status}`);
    currentConfig=await response.json();
    renderSettings();
  }catch(error){
    console.error(t('settings.error.load'),error);
    $('settings-status').textContent=t('settings.error.prefix')+error.message;
    $('settings-status').style.color='#ff6b7a';
  }
}

function renderSettings(){
  if(!currentConfig)return;
  const callable=currentConfig.callable||{};
  const defaultModel=currentConfig.default_model||'terra';
  const togglesContainer=$('callable-toggles');
  togglesContainer.innerHTML='';
  const models=['luna','spark','terra','sol','opus5','sonnet5','qwen'];
  const modelLabels={luna:t('card.luna'),spark:t('card.spark'),terra:t('card.terra'),sol:t('card.sol'),opus5:t('card.opus5'),sonnet5:t('card.sonnet5'),qwen:t('card.qwen')};
  const modelDescriptions={luna:t('model.desc.luna'),spark:t('model.desc.spark'),terra:t('model.desc.terra'),sol:t('model.desc.sol'),opus5:t('model.desc.opus5'),sonnet5:t('model.desc.sonnet5'),qwen:t('model.desc.qwen')};
  for(const model of models){
    const enabled=callable[model]!==false;
    const cooling=(currentConfig.cooldowns||{})[model];
    const item=document.createElement('div');
    item.className='toggle-item'+(enabled?'':' disabled')+(cooling?' cooling':'');
    // A cooling tier is enabled but not routable, so the switch alone is
    // misleading: the pill is what explains why traffic went elsewhere.
    const pill=cooling
      ?`<div class="cooldown-pill">${t('settings.cooling')} · ${Math.ceil(cooling.seconds/60)}m${cooling.reason?' · '+cooling.reason:''}</div>`
      :'';
    item.innerHTML=`
      <div class="toggle-label">
        <div class="toggle-name">${modelLabels[model]}</div>
        <div class="toggle-desc">${modelDescriptions[model]}</div>
        ${pill}
      </div>
      <label class="switch">
        <input type="checkbox" data-model="${model}" ${enabled?'checked':''}>
        <span class="slider"></span>
      </label>
    `;
    togglesContainer.appendChild(item);
  }
  const loadBox=$('account-load');
  if(loadBox){
    const load=currentConfig.load||{},accounts=Object.keys(load).sort((a,b)=>load[b]-load[a]||a.localeCompare(b));
    const known=['openai-codex','qwen-token'],idle=known.filter(a=>!(load[a]>0));
    const window=t('settings.load.window').replace('{n}',currentConfig.window_minutes||60);
    if(!accounts.length){
      loadBox.innerHTML=`<b>${t('settings.load.title')}</b> · ${window}<br>${t('settings.load.empty')}`;
    }else{
      const rows=accounts.map(a=>`${a} <b>${load[a]}</b>`).join(' · ');
      const idleNote=idle.length?`<br><span class="idle">${idle.join(', ')}: ${t('settings.load.idle')}</span>`:'';
      loadBox.innerHTML=`<b>${t('settings.load.title')}</b> · ${window}<br>${rows}${idleNote}<br>${t('settings.load.counts')}`;
    }
  }
  const select=$('default-model-select');
  select.innerHTML='';
  // Only a routable tier can be the orchestrator; a delegation-only target has
  // no model entry in the router and raises on the first decision.
  const orchestrators=(currentConfig.routable||[]).filter(m=>models.includes(m));
  for(const model of (orchestrators.length?orchestrators:models)){
    const option=document.createElement('option');
    option.value=model;
    option.textContent=modelLabels[model];
    if(model===defaultModel)option.selected=true;
    select.appendChild(option);
  }
}
const defaultWidths=[55,110,90,350,90,230,480],widthStore='model-router-column-widths-v3';
function saveWidths(table,cols){localStorage.setItem(widthStore,JSON.stringify({columns:cols.map(c=>parseFloat(c.style.width)),table:parseFloat(table.style.width)}))}
function initColumnResize(){const table=$('log-table'),cols=[...table.querySelectorAll('col')],heads=[...table.querySelectorAll('th')];let saved=null;try{saved=JSON.parse(localStorage.getItem(widthStore))}catch(e){}
 if(saved?.columns?.length===cols.length){saved.columns.forEach((w,i)=>cols[i].style.width=`${Math.max(55,w)}px`);table.style.width=`${Math.max(900,saved.table||saved.columns.reduce((a,b)=>a+b,0))}px`}
 heads.forEach((head,i)=>{const grip=document.createElement('div');grip.className='resizer';grip.title=t('resizer.title');head.append(grip);
  grip.addEventListener('pointerdown',e=>{e.preventDefault();grip.setPointerCapture(e.pointerId);document.body.classList.add('resizing');grip.classList.add('dragging');const startX=e.clientX,startWidth=cols[i].getBoundingClientRect().width,startTable=table.getBoundingClientRect().width;
   const move=ev=>{const width=Math.max(55,startWidth+ev.clientX-startX),delta=width-startWidth;cols[i].style.width=`${width}px`;table.style.width=`${Math.max(900,startTable+delta)}px`};
   const up=()=>{document.body.classList.remove('resizing');grip.classList.remove('dragging');grip.removeEventListener('pointermove',move);grip.removeEventListener('pointerup',up);grip.removeEventListener('pointercancel',up);saveWidths(table,cols)};
   grip.addEventListener('pointermove',move);grip.addEventListener('pointerup',up);grip.addEventListener('pointercancel',up)});
  grip.addEventListener('dblclick',()=>{cols[i].style.width=`${defaultWidths[i]}px`;table.style.width=`${defaultWidths.reduce((a,b)=>a+b,0)}px`;saveWidths(table,cols)})})}
function escText(el,text){el.textContent=text??''}
const cetFormatter=new Intl.DateTimeFormat('hu-HU',{timeZone:'Europe/Budapest',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hourCycle:'h23'});
function dateAndTime(value){const date=new Date(value);if(Number.isNaN(date.getTime()))return ['-','-'];const parts=Object.fromEntries(cetFormatter.formatToParts(date).filter(p=>p.type!=='literal').map(p=>[p.type,p.value]));return [`${parts.year}-${parts.month}-${parts.day}`,`${parts.hour}:${parts.minute}:${parts.second}`]}
function compact(vals){let out=[];for(const v of vals){let last=out[out.length-1];if(last&&last.v===v)last.n++;else out.push({v,n:1})}return out}
function filtered(){const t=$('tier').value,q=$('search').value.toLowerCase();return entries.filter(e=>(!t||e.tier===t)&&(!q||`${e.prompt_preview} ${e.turn_id} ${e.reason} ${e.model} ${(e.vetoed_by||[]).map(t=>'veto:'+t).join(' ')}`.toLowerCase().includes(q)))}
function searchFiltered(){const q=$('search').value.toLowerCase();return entries.filter(e=>!q||`${e.prompt_preview} ${e.turn_id} ${e.reason} ${e.model} ${(e.vetoed_by||[]).map(t=>'veto:'+t).join(' ')}`.toLowerCase().includes(q))}
function promptKey(e){return `${sessionIdFromTurn(e)}::${e.prompt_preview||`__missing__:${e.turn_id}`}`}
function promptGroups(list){const grouped=new Map();for(const e of list){const key=promptKey(e);if(!grouped.has(key))grouped.set(key,[]);grouped.get(key).push(e)}return [...grouped.values()]}function groups(list){return $('grouped').checked?promptGroups(list):list.map(e=>[e])}
function togglePrompt(key){setPromptExpanded(key,!isPromptExpanded(key));render()}
function routeKey(e){return e.effort?`${e.tier} · ${e.effort}`:(e.tier||'?')}function routeSummary(items){const counts=new Map();for(const item of items){const key=routeKey(item);counts.set(key,(counts.get(key)||0)+1)}return [...counts].map(([key,count])=>`${key} ×${count}`).join(' · ')}
function pill(t,label=t){const s=document.createElement('span');s.className=`pill ${t}`;s.textContent=label;return s}
function appendCell(row,text,className=''){const td=document.createElement('td');if(className)td.className=className;escText(td,text);row.append(td);return td}
function detailsTable(group){const table=document.createElement('table');table.className='details-table';const colgroup=document.createElement('colgroup');for(const width of ['110px','90px','34%','80px','150px','auto']){const col=document.createElement('col');col.style.width=width;colgroup.append(col)}table.append(colgroup);const head=document.createElement('thead'),headRow=document.createElement('tr');for(const key of ['th.date','th.time.short','th.prompt','th.calls','th.route','th.reason']){const th=document.createElement('th');if(key==='th.calls')th.className='calls';escText(th,t(key));headRow.append(th)}head.append(headRow);table.append(head);const tbody=document.createElement('tbody');for(const [index,item] of group.entries()){const row=document.createElement('tr'),[date,time]=dateAndTime(item.timestamp);appendCell(row,date);appendCell(row,time);appendCell(row,item.prompt_preview||t('not.recoverable'),'prompt-text');appendCell(row,item.api_call_count??index+1,'calls');const route=appendCell(row,'','route');route.append(pill(item.tier||'?',routeKey(item)));appendCell(row,item.reason||'?','reason');tbody.append(row)}table.append(tbody);return table}
function render(){const list=filtered(),body=$('rows');body.replaceChildren();for(const id of ['luna','spark','terra','sol','opus5','sonnet5','qwen'])$(id).textContent=list.filter(e=>e.tier===id).length;$('total').textContent=list.length;
 const gs=groups(list);if(!gs.length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=7;td.className='empty';td.textContent=t('no.entries');tr.append(td);body.append(tr);return}
 for(const g of gs){const tr=document.createElement('tr'),first=g[0],[date,time]=dateAndTime(first.timestamp),key=$('grouped').checked?promptKey(first):`${promptKey(first)}::${first.timestamp}:${first.api_call_count}`,open=isPromptExpanded(key);let td=document.createElement('td');td.className='expand';const toggle=document.createElement('button');toggle.type='button';toggle.className='prompt-toggle';toggle.textContent=open?'▾':'▸';toggle.title=open?t('details.close'):t('details.open');toggle.addEventListener('click',()=>togglePrompt(key));td.append(toggle);tr.append(td);appendCell(tr,date);appendCell(tr,time);td=appendCell(tr,first.prompt_preview||t('not.recoverable'));td.title=first.prompt_preview||t('related.missing');appendCell(tr,g.length,'calls');td=document.createElement('td');td.className='route';for(const [i,r] of compact(g.map(routeKey)).entries()){if(i)td.append(' → ');td.append(pill(r.v.split(' · ')[0],r.v));if(r.n>1)td.append(`×${r.n}`)}tr.append(td);appendCell(tr,compact(g.map(x=>x.reason)).map(r=>r.v+(r.n>1?` ×${r.n}`:'')).join(' → '),'reason');body.append(tr);if(open){const detail=document.createElement('tr'),detailCell=document.createElement('td');detail.className='detail';detailCell.colSpan=7;detailCell.append(detailsTable(g));detail.append(detailCell);body.append(detail)}}}
function renderAgents(activity){const s=activity.summary||{},tree=$('agent-tree');$('agent-summary').textContent=`${s.running||0} ${t('agents.sum.running')} · ${s.completed||0} ${t('agents.sum.completed')} · ${s.failed||0} ${t('agents.sum.failed')}`;tree.replaceChildren();const finished=[];for(const parent of activity.parents||[]){const active=(parent.children||[]).filter(c=>c.state==='running');finished.push(...(parent.children||[]).filter(c=>c.state!=='running'));if(!active.length)continue;const wrap=document.createElement('div');wrap.className='agent-parent';const title=document.createElement('div');title.className='agent-session';title.textContent=`${t('agents.main.agent')} · ${parent.session_id}`;wrap.append(title);for(const child of active){const row=document.createElement('div');row.className='agent-child';const dot=document.createElement('span');dot.className='agent-dot running';const text=document.createElement('div');const goal=document.createElement('div');goal.className='agent-goal';goal.textContent=child.goal;const phase=document.createElement('div');phase.className='agent-activity';phase.textContent=child.activity;text.append(goal,phase);const meta=document.createElement('div');meta.className='agent-meta';meta.textContent=`${t('state.running.short')} · ${child.age_seconds}s${child.model?`\n${child.model}`:''}`;row.append(dot,text,meta);wrap.append(row);const consoleBox=document.createElement('details');consoleBox.className='console';consoleBox.open=true;const summary=document.createElement('summary');summary.textContent=`${t('agents.close.last')} ${Math.min((child.console||[]).length,10)} ${t('agents.close.steps')}`;consoleBox.append(summary);const events=child.console||[];if(!events.length){const waiting=document.createElement('div');waiting.className='console-empty';waiting.textContent=t('agents.no.events');consoleBox.append(waiting)}for(const event of events){const eventEl=document.createElement('div');eventEl.className='console-event compact';eventEl.textContent=`${new Date(event.timestamp*1000).toLocaleTimeString('hu-HU')} · ${event.tool}`;consoleBox.append(eventEl)}wrap.append(consoleBox)}tree.append(wrap)}if(finished.length){const history=document.createElement('details');history.className='agent-history';const summary=document.createElement('summary');summary.textContent=`${t('agents.recently.done')} (${finished.length})`;history.append(summary);for(const child of finished.slice(0,20)){const item=document.createElement('div');item.className='history-item';item.textContent=`${child.goal} · ${child.age_seconds}s`;history.append(item)}tree.append(history)}if(!tree.childElementCount){const empty=document.createElement('div');empty.className='agent-empty';empty.textContent=t('agents.none.active');tree.append(empty)}}
function setWordWrap(){const enabled=$('word-wrap').checked;$('log-table').classList.toggle('word-wrap',enabled);localStorage.setItem('model-router-word-wrap',enabled?'1':'0')}
document.querySelectorAll('.tab').forEach(tab=>tab.addEventListener('click',()=>setTab(tab.dataset.tab)));for(const id of ['tier','search','grouped'])$(id).addEventListener('input',render);$('word-wrap').checked=localStorage.getItem('model-router-word-wrap')==='1';$('word-wrap').addEventListener('input',setWordWrap);setWordWrap();initColumnResize();setTab(selectedTab||'router',false); applyLanguage();

// Settings event listeners
document.getElementById('callable-toggles').addEventListener('change',async(e)=>{
  if(e.target.type!=='checkbox'||!currentConfig)return;
  const model=e.target.dataset.model;
  const enabled=e.target.checked;
  currentConfig.callable[model]=enabled;
  e.target.closest('.toggle-item').classList.toggle('disabled',!enabled);
  await saveSettings();
});
document.getElementById('default-model-select').addEventListener('change',async(e)=>{
  if(!currentConfig)return;
  currentConfig.default_model=e.target.value;
  await saveSettings();
});
document.getElementById('language-select').addEventListener('change',async(e)=>{
  currentLang=e.target.value;
  localStorage.setItem('model-router-lang',currentLang);
  applyLanguage();
  // applyLanguage only walks [data-i18n] nodes; without this the tables and
  // agent cards the renderers already built stay in the previous language.
  if(typeof render==='function')render();
  if(typeof renderAgents==='function'&&typeof agentActivity!=='undefined')renderAgents(agentActivity);
});
// Initialize language select value on load
document.getElementById('language-select').value=currentLang;
async function saveSettings(){
  if(!currentConfig)return;
  const statusEl=$('settings-status');
  statusEl.textContent=t('settings.saving');
  statusEl.style.color='#c4b5fd';
  try{
    const response=await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({callable:currentConfig.callable,default_model:currentConfig.default_model})});
    if(!response.ok)throw new Error(`HTTP ${response.status}`);
    const result=await response.json();
    if(result.success){
      statusEl.textContent=t('settings.saved');
      statusEl.style.color='#88e36f';
    }else{
      throw new Error(result.error||t('settings.error.unknown'));
    }
  }catch(error){
    statusEl.textContent=t('settings.error.prefix')+error.message;
    statusEl.style.color='#ff6b7a';
    console.error(t('settings.error.save'),error);
  }
}
function formatDuration(seconds){seconds=Math.max(0,Number(seconds)||0);if(seconds<60)return `${seconds} mp`;const minutes=Math.floor(seconds/60),rest=seconds%60;return minutes<60?`${minutes} ${t('duration.m')} ${rest} ${t('duration.s')}`:`${Math.floor(minutes/60)} ${t('duration.h')} ${minutes%60} ${t('duration.m')}`}
function agentRunParts(timestamp){if(!timestamp)return ['—','—'];return dateAndTime(new Date(Number(timestamp)*1000).toISOString())}
function renderAgents(activity){const s=activity.summary||{},tree=$('agent-tree');$('agent-summary').textContent=`${s.running||0} ${t('agents.sum.running')} · ${s.completed||0} ${t('agents.sum.completed')} · ${s.failed||0} ${t('agents.sum.failed')}`;tree.replaceChildren();const allParents=activity.parents||[],parents=allParents.slice(0,12);$('agent-summary').textContent=`${s.running||0} ${t('agents.sum.running')} · ${s.completed||0} ${t('agents.sum.completed')} · ${s.failed||0} ${t('agents.sum.failed')} · ${parents.length}/${allParents.length} ${t('agents.recent')}`;if(!parents.length){const empty=document.createElement('div');empty.className='agent-empty';empty.textContent=t('agents.none');tree.append(empty);return}for(const parent of parents){const branch=document.createElement('article');branch.className='agent-branch';const header=document.createElement('div');header.className='agent-parent-head';const label=document.createElement('div');label.className='agent-session';label.textContent=t('agents.main.thread');const id=document.createElement('code');id.textContent=parent.session_id;header.append(label,id);const prompt=document.createElement('div');prompt.className='agent-parent-prompt';prompt.textContent=parent.prompt||t('agents.prev.task');branch.append(header,prompt);const children=document.createElement('div');children.className='agent-children';for(const child of parent.children||[]){const row=document.createElement('div');row.className='agent-child-row';const stem=document.createElement('span');stem.className=`agent-dot ${child.state==='running'?'running':''}`;const body=document.createElement('div');body.className='agent-child-body';const reason=document.createElement('div');reason.className='agent-reason';reason.textContent=child.reason||t('agents.reason.default');body.append(reason);const meta=document.createElement('div');meta.className='agent-meta';meta.textContent=`${child.state==='running'?t('state.running.short'):t('state.done.short')} · ${formatDuration(child.age_seconds)}${child.api_calls?` · ${child.api_calls} ${t('agents.calls')}`:''}${child.model?`\n${child.model}`:''}`;row.append(stem,body,meta);children.append(row)}branch.append(children);tree.append(branch)}}
const agentExpansionKey='ai-home-lab-agent-main-expansion-v1';function loadAgentExpansion(){try{return JSON.parse(localStorage.getItem(agentExpansionKey)||'{}')}catch(e){return {}}}function saveAgentExpansion(state){localStorage.setItem(agentExpansionKey,JSON.stringify(state))}renderAgents=function(activity){const s=activity.summary||{},tree=$('agent-tree'),allParents=activity.parents||[],parents=allParents.slice(0,12),expansion=loadAgentExpansion();$('agent-summary').textContent=`${s.running||0} ${t('agents.sum.running')} · ${s.completed||0} ${t('agents.sum.completed')} · ${s.failed||0} ${t('agents.sum.failed')} · ${parents.length}/${allParents.length} ${t('agents.recent')}`;tree.replaceChildren();if(!parents.length){const empty=document.createElement('div');empty.className='agent-empty';empty.textContent=t('agents.none');tree.append(empty);return}for(const parent of parents){const running=(parent.children||[]).some(child=>child.state==='running'),storedOpen=Object.hasOwn(expansion,parent.session_id)&&!!expansion[parent.session_id],open=running||storedOpen;const card=document.createElement('article');card.className=`agent-main-card ${open?'is-open':'is-closed'}`;const header=document.createElement('button');header.type='button';header.className='agent-main-toggle';header.setAttribute('aria-expanded',String(open));const [runDate,runTime]=agentRunParts(parent.started_at);const date=document.createElement('span');date.className='agent-main-date';date.textContent=runDate;const time=document.createElement('span');time.className='agent-main-time';time.textContent=runTime;const heading=document.createElement('div');heading.className='agent-main-heading';const preview=document.createElement('span');preview.className='agent-main-preview';preview.textContent=parent.prompt||t('agents.prev.task');heading.append(preview);const stats=document.createElement('div');stats.className='agent-main-stats';const count=document.createElement('span');count.className='agent-stat';count.textContent=`${(parent.children||[]).length} ${t('agents.subtasks')}`;const status=document.createElement('span');status.className=`agent-stat ${running?'running':''}`;status.textContent=running?t('agents.running'):t('agents.done');const chevron=document.createElement('span');chevron.className='agent-chevron';chevron.textContent=open?'⌃':'⌄';stats.append(count,status,chevron);header.append(date,time,heading,stats);header.addEventListener('click',()=>{const next=loadAgentExpansion();next[parent.session_id]=!open;saveAgentExpansion(next);renderAgents(activity)});card.append(header);const detail=document.createElement('div');detail.className='agent-main-detail';detail.hidden=!open;const prompt=document.createElement('div');prompt.className='agent-full-prompt';prompt.textContent=parent.prompt||t('agents.prev.task');detail.append(prompt);const workers=document.createElement('div');workers.className='agent-worker-grid';for(const child of parent.children||[]){const worker=document.createElement('div');worker.className=`agent-worker-card ${child.state==='running'?'running':''}`;const top=document.createElement('div');top.className='agent-worker-top';const purpose=document.createElement('span');purpose.className='agent-worker-purpose';purpose.textContent=child.reason||t('agents.reason.default');const access=child.access_mode==='requested_read_only'?document.createElement('span'):null;if(access){access.className='agent-read-only';access.textContent=t('agents.requested.ro');access.title=t('agents.requested.ro.title')}const state=document.createElement('span');state.className=`agent-worker-state ${child.state==='running'?'running':''}`;state.textContent=child.state==='running'?t('state.running.short'):t('state.done.short');if(access)top.append(purpose,access,state);else top.append(purpose,state);const details=document.createElement('div');details.className='agent-worker-details';details.textContent=`${formatDuration(child.age_seconds)} · ${child.api_calls||0} ${t('agents.calls')}`;const model=document.createElement('div');model.className='agent-worker-model';model.textContent=child.model||t('agents.unknown.model');worker.append(top,details,model);workers.append(worker)}detail.append(workers);card.append(detail);tree.append(card)}};
let agentActivity={parents:[]};function childSessionIds(){return new Set((agentActivity.parents||[]).flatMap(parent=>(parent.children||[]).map(child=>child.agent_session_id).filter(Boolean)))}function turnBelongsToChild(entry){const turn=String(entry.turn_id||'');return [...childSessionIds()].some(sessionId=>turn.startsWith(`${sessionId}:`))}const syntheticLifecyclePrefixes=['[ASYNC DELEGATION BATCH COMPLETE','[ASYNC DELEGATION COMPLETE','[Your active task list was preserved across context compression]','[IMPORTANT: Background process ','Review the conversation above and consider saving to memory if appropriate.','[CONTEXT COMPACTION'];function isSyntheticLifecyclePrompt(entry){const prompt=String(entry.prompt_preview||''),turn=String(entry.turn_id||'');if(entry&&entry.is_internal_prompt===true)return true;return turn.includes(':sa-')||syntheticLifecyclePrefixes.some(prefix=>prompt.startsWith(prefix))}function childEntries(child,list,seen=new Set()){const sessionId=child?.agent_session_id;if(!sessionId||seen.has(sessionId))return [];seen.add(sessionId);const own=(child.routed_calls&&child.routed_calls.length)?child.routed_calls:list.filter(entry=>String(entry.turn_id||'').startsWith(`${sessionId}:`));const nested=(agentActivity.parents||[]).find(parent=>parent.session_id===sessionId);const descendants=(nested?.children||[]).flatMap(next=>childEntries(next,list,seen));return [...own,...descendants]}function visibleEntries(list){const bridgeRuns=new Set(agentActivity.external_bridge_run_ids||[]);return list.filter(entry=>!bridgeRuns.has(sessionIdFromTurn(entry))&&!turnBelongsToChild(entry)&&!isSyntheticLifecyclePrompt(entry))}function sessionIdFromTurn(entry){return String(entry?.turn_id||'').split(':')[0]}function systemEntriesForRoot(root,list){const rootTurnId=String(root?.turn_id||''),sessionId=sessionIdFromTurn(root),allRoots=visibleEntries(list).sort((a,b)=>Date.parse(a.timestamp)-Date.parse(b.timestamp)),sessionRoots=allRoots.filter(entry=>sessionIdFromTurn(entry)===sessionId);return list.filter(entry=>{if(!isSyntheticLifecyclePrompt(entry))return false;const directParent=String(entry.parent_turn_id||'');if(directParent){return directParent===rootTurnId}if(sessionIdFromTurn(entry)!==sessionId)return false;const when=Date.parse(entry.timestamp),preceding=sessionRoots.filter(candidate=>Date.parse(candidate.timestamp)<=when).at(-1);return preceding&&String(preceding.turn_id||'')===rootTurnId})}function promptsMatch(left,right){const a=String(left||'').trim(),b=String(right||'').trim();return a&&b&&(a===b||a.startsWith(b)||b.startsWith(a))}function nearestParent(parents,timestamp){const target=Date.parse(timestamp)/1000;if(!Number.isFinite(target))return parents[0]||null;return parents.reduce((best,parent)=>!best||Math.abs(Number(parent.started_at)-target)<Math.abs(Number(best.started_at)-target)?parent:best,null)}function parentForGroup(group){for(const entry of group){const sessionId=sessionIdFromTurn(entry);const sessionParents=(agentActivity.parents||[]).filter(parent=>parent.session_id===sessionId);if(!sessionParents.length)continue;const prompt=String(entry.prompt_preview||'').trim();const promptMatch=sessionParents.find(parent=>promptsMatch(prompt,parent.prompt));if(promptMatch)return promptMatch;return null}return null}function groupRunState(group){const parent=parentForGroup(group);if(!parent)return 0;return (parent.children||[]).some(child=>child.state==='running')?1:0}function agentDetails(parent){const grid=document.createElement('div');grid.className='agent-worker-grid router-agent-workers';for(const child of parent.children||[]){const worker=document.createElement('div');worker.className=`agent-worker-card ${child.state==='running'?'running':''}`;const top=document.createElement('div');top.className='agent-worker-top';const purpose=document.createElement('span');purpose.className='agent-worker-purpose';purpose.textContent=child.reason||t('agents.reason.default');const state=document.createElement('span');state.className=`agent-worker-state ${child.state==='running'?'running':''}`;state.textContent=child.state==='running'?t('state.running.short'):t('state.done.short');top.append(purpose,state);if(child.access_mode==='requested_read_only'){const access=document.createElement('span');access.className='agent-read-only';access.textContent=t('agents.requested.ro');access.title=t('agents.requested.ro.title');top.append(access)}const details=document.createElement('div');details.className='agent-worker-details';details.textContent=`${formatDuration(child.age_seconds)} · ${child.api_calls||0} ${t('agents.calls')}`;const model=document.createElement('div');model.className='agent-worker-model';model.textContent=child.model||t('agents.unknown.model');worker.append(top,details,model);grid.append(worker)}return grid}function nestedParentForChild(child){const descendants=child.children||[],hasDescendants=child.children?.length;return hasDescendants?{children:descendants}:null}function nodeRoute(calls){return calls.length?routeSummary(calls):t('exec.no.route')}const taskTreeExpansionKey='model-router-task-tree-expansion-v2';function taskTreeExpansions(){try{return JSON.parse(localStorage.getItem(taskTreeExpansionKey)||'{}')}catch(e){return {}}}function taskTreeOpen(key,defaultOpen=true){const state=taskTreeExpansions();return Object.hasOwn(state,key)?!!state[key]:defaultOpen}function setTaskTreeOpen(key,open){const state=taskTreeExpansions();state[key]=open;localStorage.setItem(taskTreeExpansionKey,JSON.stringify(state))}function appendTaskTreeNode(tree,node,depth){const children=node.children||[],hasChildren=children.length>0,open=hasChildren&&taskTreeOpen(node.id,true),row=document.createElement('div');row.className=`task-tree-row ${node.state==='running'?'running':''}`;row.style.setProperty('--tree-depth',depth);row.setAttribute('role','treeitem');row.setAttribute('aria-level',String(depth+1));if(hasChildren)row.setAttribute('aria-expanded',String(open));const branch=document.createElement(hasChildren?'button':'span');branch.className=hasChildren?'task-tree-toggle':'task-tree-branch';branch.textContent=hasChildren?(open?'▾':'▸'):'•';if(hasChildren){branch.type='button';branch.title=open?t('agents.close.tasks'):t('agents.open.tasks');branch.addEventListener('click',()=>{setTaskTreeOpen(node.id,!open);render()})}const kind=document.createElement('span');kind.className='task-tree-kind';kind.textContent=node.kind;const description=document.createElement('span');description.className='task-tree-description';description.textContent=node.description||t('agents.no.desc');description.title=description.textContent;const state=document.createElement('span');state.className=`task-tree-state ${node.state==='running'?'running':''}`;state.textContent=node.state==='running'?t('state.running.short'):t('state.done.short');row.append(branch,kind,description,state);tree.append(row);if(open)for(const child of children)appendTaskTreeNode(tree,child,depth+1)}function childTaskNode(child){return {id:child.id||child.agent_session_id||child.goal,kind:child.model?.toUpperCase()||'WORKER',description:child.task_description||child.goal,state:child.state,children:(child.children||[]).map(childTaskNode)}}function lifecycleTaskNodes(systemCalls,parentId){return systemCalls.map((entry,index)=>({id:`${parentId}:lifecycle:${index}`,kind:'LIFECYCLE',description:t('internal.router.step'),state:'completed',children:[]}))}function promptExecutionTree(group,systemCalls,parent){const tree=document.createElement('div');tree.className='task-tree';tree.setAttribute('role','tree');const rootId=`${parent?.session_id||sessionIdFromTurn(group?.[0])}:root`;for(const node of [...lifecycleTaskNodes(systemCalls,rootId),...(parent?.children||[]).map(childTaskNode)])appendTaskTreeNode(tree,node,0);return tree}function legacySystem(body,systemCalls,sessionId){const key=`__system__:${sessionId}`,open=isPromptExpanded(key),first=systemCalls[0],[date,time]=dateAndTime(first.timestamp),tr=document.createElement('tr'),expand=document.createElement('td');expand.className='expand';const toggle=document.createElement('button');toggle.type='button';toggle.className='prompt-toggle';toggle.textContent=open?'▾':'▸';toggle.title=open?t('close.router.steps'):t('open.router.steps');toggle.setAttribute('aria-expanded',String(open));toggle.addEventListener('click',()=>{setPromptExpanded(key,!open);render()});expand.append(toggle);tr.append(expand);appendCell(tr,date);appendCell(tr,time);const prompt=appendCell(tr,t('root.continuation'));prompt.title=t('root.continuation.title');appendCell(tr,systemCalls.length,'calls');const route=document.createElement('td');route.className='route';for(const [index,item] of compact(systemCalls.map(routeKey)).entries()){if(index)route.append(' → ');route.append(pill(item.v.split(' · ')[0],item.v));if(item.n>1)route.append(`×${item.n}`)}tr.append(route);appendCell(tr,t('root.continuation.reason'),'reason');body.append(tr);if(open){const detail=document.createElement('tr'),cell=document.createElement('td');detail.className='detail';cell.colSpan=7;cell.append(detailsTable(systemCalls));detail.append(cell);body.append(detail)}}
render=function(){const allEntries=filtered(),list=visibleEntries(allEntries),body=$('rows'),shownSystemSessions=new Set();body.replaceChildren();for(const id of ['luna','spark','terra','sol','opus5','sonnet5'])$(id).textContent=allEntries.filter(entry=>entry.tier===id).length;$('total').textContent=allEntries.length;const gs=groups(list).sort((left,right)=>groupRunState(left)-groupRunState(right));if(!gs.length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=7;td.className='empty';td.textContent=t('no.entries');tr.append(td);body.append(tr);return}for(const group of gs){const first=group[0],parent=parentForGroup(group),key=promptKey(first),open=isPromptExpanded(key),childRecords=(parent?.children||[]).flatMap(child=>childEntries(child,allEntries)),childCalls=childRecords.length,systemCalls=systemEntriesForRoot(first,allEntries),tr=document.createElement('tr'),[date,time]=dateAndTime(first.timestamp);const hasDetails=systemCalls.length>0||childCalls>0||(parent?.children||[]).length>0;const expand=document.createElement('td');expand.className='expand';if(hasDetails){const toggle=document.createElement('button');toggle.type='button';toggle.className='prompt-toggle';toggle.textContent=open?'▾':'▸';toggle.title=open?t('subagent.close'):t('subagent.open');toggle.addEventListener('click',()=>{setPromptExpanded(key,!open);render()});expand.append(toggle)}tr.append(expand);appendCell(tr,date);appendCell(tr,time);const prompt=appendCell(tr,first.prompt_preview||t('not.recoverable'),'prompt-text');prompt.title=first.prompt_preview||'';const runningChildren=(parent?.children||[]).filter(child=>child.state==='running');const activeMainTurns=(agentActivity.active_turns||[]).filter(turn=>group.some(entry=>String(entry.turn_id||'')===String(turn.turn_id||'')));if(activeMainTurns.length){tr.classList.add('active-router-row');const play=document.createElement('span');play.className='play-indicator';play.textContent='▶';play.title=t('play.running.agent');prompt.append(play);const live=document.createElement('span');live.className='running-agent-pill';live.textContent=`${t('state.running.short')} · ${activeMainTurns.map(turn=>turn.model||t('label.model.short')).join(', ')}`;live.title=`${t('label.active.turn')}${activeMainTurns.map(turn=>turn.turn_id).join(', ')}`;prompt.append(live)}if(runningChildren.length){tr.classList.add('active-router-row');const play=document.createElement('span');play.className='play-indicator';play.textContent='▶';play.title=t('play.running.subagent');prompt.append(play);const live=document.createElement('span');live.className='running-agent-pill';live.textContent=`${t('agents.pill.running')} · ${runningChildren.length} ${t('agents.pill.agent')}`;prompt.append(live)}appendCell(tr,group.length,'calls');const route=document.createElement('td');route.className='route';for(const [index,item] of compact(group.map(routeKey)).entries()){if(index)route.append(' → ');route.append(pill(item.v.split(' · ')[0],item.v));if(item.n>1)route.append(`×${item.n}`)}tr.append(route);appendCell(tr,runningChildren.length?t('subagent.running'):parent?t('agents.done'):t('main.agent'),'reason');body.append(tr);if(open){const detail=document.createElement('tr'),cell=document.createElement('td');detail.className='detail';cell.colSpan=7;cell.append(promptExecutionTree(group,systemCalls,parent));detail.append(cell);body.append(detail)}}};loadAgents=async function(){try{const response=await fetch('/api/agents',{cache:'no-store'});if(response.ok){agentActivity=await response.json();render()}}catch(e){agentActivity={parents:[]};render()}};loadAgents();
/* Reference execution-tree renderer: one compact router header and one tree panel. */
function executionRouteKey(call){const tier=String(call?.tier||call?.model||'?').toLowerCase(),effort=String(call?.effort||'').toLowerCase();return effort?`${tier} · ${effort}`:tier}
function executionOwnCalls(node){const raw=Array.isArray(node?.routed_calls)?node.routed_calls:[];return raw.length?raw:[]}
function executionOwnCount(node){const raw=executionOwnCalls(node);return raw.length||Number(node?.api_calls)||0}
function executionCalls(node){return [...executionOwnCalls(node),...(node.children||[]).flatMap(executionCalls)]}
function executionTotalCalls(node){const raw=executionCalls(node);return raw.length||executionOwnCount(node)+(node.children||[]).reduce((sum,child)=>sum+executionTotalCalls(child),0)}
function executionRawCall(node){const calls=executionOwnCalls(node);return calls.length?calls[calls.length-1]:null}
function executionKind(node){const raw=executionRawCall(node),source=`${node.kind||''} ${raw?.tier||raw?.model||node.model||''}`.toLowerCase();if(node.kind==='LIFECYCLE')return 'lifecycle';if(source.includes('qwen'))return 'qwen';if(source.includes('opus5')||source.includes('claude-opus-5'))return 'opus5';if(source.includes('sonnet5')||source.includes('claude-sonnet-5'))return 'sonnet5';if(source.includes('terra'))return 'terra';if(source.includes('spark'))return 'spark';if(source.includes('luna'))return 'luna';if(source.includes('sol'))return 'sol';return 'worker'}
function executionModelLabel(node){if(node.kind==='LIFECYCLE')return 'LIFECYCLE';const raw=executionRawCall(node),kind=executionKind(node);return String(raw?.tier||raw?.model||node.model||kind).toUpperCase()}
function executionNode(child){return {id:child.bridge_run_id||child.id||child.agent_session_id||child.goal,bridge_run_id:child.bridge_run_id,model:child.model,goal:child.goal,task_description:child.task_description||child.goal,state:child.state,external:!!child.external,access_mode:child.access_mode,metrics:child.metrics||{},routed_calls:child.routed_calls||[],api_calls:child.api_calls,children:(child.children||[]).map(executionNode)}}
function lifecycleDescription(entry){const completion=/^\[(?:ASYNC DELEGATION|CONTEXT COMPACTION|Your active task list)/i;for(const value of [entry?.lifecycle_prompt,entry?.task_description,entry?.task,entry?.first_user_message,entry?.user_prompt,entry?.prompt_preview]){const text=String(value||'').trim();if(text&&!completion.test(text))return text}return t('internal.router.step')}
function executionLifecycleNodes(systemCalls,parentId){return systemCalls.map((entry,index)=>({id:`${parentId}:lifecycle:${index}`,kind:'LIFECYCLE',task_description:lifecycleDescription(entry),state:'completed',routed_calls:[entry],api_calls:1,children:[]}))}
function executionCallTier(call){return String(call?.tier||call?.model||'?').toLowerCase()}
function executionFilteredNode(node,tier=''){const children=(node.children||[]).map(child=>executionFilteredNode(child,tier)).filter(Boolean),routed_calls=executionOwnCalls(node).filter(call=>!tier||executionCallTier(call)===tier);if(tier&&!routed_calls.length&&!children.length)return null;return {...node,routed_calls,api_calls:tier?routed_calls.length:node.api_calls,children}}
function executionSummary(runCalls){const summary={total:0,luna:0,spark:0,terra:0,sol:0,opus5:0,sonnet5:0,qwen:0};for(const call of (runCalls||[]).flat()){const tier=executionCallTier(call);summary.total++;if(Object.hasOwn(summary,tier))summary[tier]++}return summary}
function appendRoutePills(row,calls){const counts=new Map();for(const call of calls||[]){const key=executionRouteKey(call);counts.set(key,(counts.get(key)||0)+1)}for(const [key,count] of counts){const [tier]=key.split(' · ');row.append(pill(tier,`${key} ×${count}`))}}
function appendExecutionNode(tree,node,depth,siblings=[]){const children=node.children||[],hasChildren=children.length>0,open=hasChildren&&taskTreeOpen(node.id,true),kind=executionKind(node),siblingIndex=siblings.indexOf(node),hasPreviousSibling=siblingIndex>0,hasNextSibling=siblingIndex>=0&&siblingIndex<siblings.length-1,row=document.createElement('div');row.className=`execution-tree-row task-tree-${kind} ${node.state==='running'?'running':''} ${hasPreviousSibling?'task-tree-has-prev-sibling':''} ${hasNextSibling?'task-tree-has-next-sibling':''}`;row.style.setProperty('--tree-depth',depth);row.setAttribute('role','treeitem');row.setAttribute('aria-level',String(depth+1));if(hasChildren)row.setAttribute('aria-expanded',String(open));const disclosure=document.createElement(hasChildren?'button':'span');disclosure.className=hasChildren?'execution-tree-toggle':'execution-tree-spacer';disclosure.textContent=hasChildren?(open?'▾':'▸'):'';if(hasChildren){disclosure.type='button';disclosure.title=open?t('agents.close.inner'):t('agents.open.inner');disclosure.addEventListener('click',event=>{event.stopPropagation();setTaskTreeOpen(node.id,!open);render()})}const marker=document.createElement('span');marker.className=`task-tree-marker ${kind}`;marker.setAttribute('aria-hidden','true');const model=document.createElement('span');model.className='execution-tree-model';model.textContent=executionModelLabel(node);const description=document.createElement('span');description.className='execution-tree-description';description.textContent=node.task_description||node.goal||t('agents.no.desc');description.title=description.textContent;if(node.external){const badges=[];badges.push(t('exec.external'));if(node.access_mode==='read_only')badges.push(t('exec.read.only'));else if(node.access_mode==='requested_read_only')badges.push(t('agents.requested.ro'));description.textContent=`${node.goal||t('exec.opus.review')} · ${badges.join(' · ')} · ${description.textContent}`}const accounting=document.createElement('span');accounting.className='execution-tree-accounting';const ownCalls=executionOwnCount(node),totalCalls=executionTotalCalls(node),metrics=node.metrics||{};accounting.textContent=node.external?`turns ${ownCalls} · input ${metrics.input_tokens??'—'} · output ${metrics.output_tokens??'—'} · cache ${metrics.cache_read_input_tokens??'—'} · cost ${metrics.total_cost_usd??'—'} · duration ${metrics.duration_seconds??'—'}s`:`${t('exec.own')} ${ownCalls} · ${t('exec.total')} ${totalCalls}`;const routes=document.createElement('span');routes.className='execution-tree-routes';appendRoutePills(routes,executionCalls(node));const state=document.createElement('span'),stateLabels={running:t('state.running.short'),success:t('state.success.short'),error:t('state.error.short'),timeout:'TIMEOUT','max-turn':'MAX-TURN',budget:'BUDGET',completed:t('state.done.short')};state.className=`execution-tree-state ${node.state==='running'?'running':''}`;state.textContent=stateLabels[node.state]||String(node.state||t('state.done.short')).toUpperCase();row.append(disclosure,marker,model,description,accounting,routes,state);tree.append(row);if(open&&hasChildren){const childrenEl=document.createElement('div');childrenEl.className='execution-tree-children';childrenEl.style.setProperty('--tree-depth',depth);for(const child of children)appendExecutionNode(childrenEl,child,depth+1,children);tree.append(childrenEl)}}
function executionScope(group,systemCalls,parent,tier=''){const rootId=`${parent?.session_id||sessionIdFromTurn(group?.[0])}:root`,rawNodes=[...executionLifecycleNodes(systemCalls,rootId),...(parent?.children||[]).map(executionNode)],nodes=rawNodes.map(node=>executionFilteredNode(node,tier)).filter(Boolean),routed_calls=(group||[]).filter(call=>!tier||executionCallTier(call)===tier),root={routed_calls,children:nodes};return {calls:executionCalls(root),nodes}}
function buildEntryIndex(allEntries){const childIds=childSessionIds(),bySession=new Map(),groupsMap=new Map(),lifecycleByTurn=new Map();for(const entry of allEntries){const session=sessionIdFromTurn(entry);if(!bySession.has(session))bySession.set(session,[]);bySession.get(session).push(entry);if(isSyntheticLifecyclePrompt(entry)){const parent=String(entry.parent_turn_id||'');if(parent){if(!lifecycleByTurn.has(parent))lifecycleByTurn.set(parent,[]);lifecycleByTurn.get(parent).push(entry)}continue}if(childIds.has(session))continue;const key=promptKey(entry);if(!groupsMap.has(key))groupsMap.set(key,[]);groupsMap.get(key).push(entry)}return {bySession,groups:[...groupsMap.values()],lifecycleByTurn}}
function childLogEntries(child,index,seen=new Set()){const sessionId=child?.agent_session_id;if(!sessionId||seen.has(sessionId))return [];seen.add(sessionId);const own=index.bySession.get(sessionId)||[],nested=(agentActivity.parents||[]).find(parent=>parent.session_id===sessionId),descendants=(nested?.children||[]).flatMap(next=>childLogEntries(next,index,seen));return [...own,...descendants]}
function indexedSystemEntries(group,index){const direct=[...new Set(group.flatMap(entry=>index.lifecycleByTurn.get(String(entry.turn_id||''))||[]))];if(direct.length)return direct;return systemEntriesForRoot(group[0],index.bySession.get(sessionIdFromTurn(group[0]))||[])}
function rootRunRecords(allEntries,tier=''){const index=buildEntryIndex(allEntries);return index.groups.map(group=>{const first=group[0],parent=parentForGroup(group),systemCalls=indexedSystemEntries(group,index),scope=executionScope(group,systemCalls,parent,tier),childLogs=(parent?.children||[]).flatMap(child=>childLogEntries(child,index));return {group,first,parent,systemCalls,scope,rawEntries:[...new Set([...group,...systemCalls,...childLogs])]}}).filter(run=>!tier||run.scope.calls.length)}
function limitRootRuns(runs,limit){const count=Number(limit);return runs.slice(-Math.max(0,Number.isFinite(count)?count:20))}
function rawEntryKey(entry){return entry?.id||`${entry?.timestamp||''}::${entry?.turn_id||''}::${entry?.api_call_count||''}::${entry?.tier||''}::${entry?.model||''}`}
function rawEntriesForRootRuns(runs,allEntries){const selected=new Set(runs.flatMap(run=>run.rawEntries||[]).map(rawEntryKey));return allEntries.filter(entry=>selected.has(rawEntryKey(entry)))}
function executionTree(nodes){const tree=document.createElement('div');tree.className='execution-tree';tree.setAttribute('role','tree');for(const node of nodes||[])appendExecutionNode(tree,node,0,nodes);return tree}
function executionState(group,parent){const active=(agentActivity.active_turns||[]).some(turn=>group.some(entry=>String(entry.turn_id||'')===String(turn.turn_id||'')))||(parent?.children||[]).some(child=>child.state==='running');return active?'running':'done'}
setWordWrap=function(){const enabled=$('word-wrap').checked;$('log-table').classList.toggle('word-wrap',enabled);$('runs').classList.toggle('word-wrap',enabled);localStorage.setItem('model-router-word-wrap',enabled?'1':'0')};$('word-wrap').addEventListener('input',setWordWrap);setWordWrap();
render=function(){const allEntries=searchFiltered(),groupedMode=$('grouped').checked,runs=$('runs'),tier=$('tier').value,rootRuns=rootRunRecords(allEntries,tier),limitedRoots=limitRootRuns(rootRuns,$('last').value),displayRoots=[...limitedRoots].sort((left,right)=>groupRunState(left.group)-groupRunState(right.group)),rawList=rawEntriesForRootRuns(limitedRoots,allEntries);runs.replaceChildren();const runData=groupedMode?displayRoots:rawList.map(group=>{const first=group,parent=null,systemCalls=[],scope=executionScope([group],systemCalls,parent,tier);return {group:[group],first,parent,systemCalls,scope}}).filter(run=>!tier||run.scope.calls.length);$('status').textContent=`${limitedRoots.length} ${t('status.rootprompts')} · ${new Date().toLocaleTimeString(t('status.locale'))}`;const summary=executionSummary(runData.map(run=>run.scope.calls));for(const id of ['luna','spark','terra','sol','opus5','sonnet5','qwen'])$(id).textContent=summary[id];$('total').textContent=summary.total;if(!runData.length){const empty=document.createElement('div');empty.className='empty';empty.textContent=t('no.entries');runs.append(empty);return}for(const record of runData){const {group,first,parent,scope}=record,key=groupedMode?promptKey(first):`${promptKey(first)}::${first.timestamp}:${first.api_call_count}`,open=isPromptExpanded(key),hasDetails=scope.nodes.length>0,accountingCalls=scope.calls,workerCalls=scope.nodes.filter(node=>node.kind!=='LIFECYCLE').flatMap(executionCalls).length,state=executionState(group,parent),[date,time]=dateAndTime(first.timestamp),run=document.createElement('article');run.className=`router-run ${open?'is-open':'is-closed'} ${state==='running'?'running':''}`;const header=document.createElement(hasDetails?'button':'div');if(hasDetails)header.type='button';header.className=`router-run-header ${hasDetails?'':'no-details'}`;if(hasDetails)header.setAttribute('aria-expanded',String(open));if(hasDetails){const toggle=document.createElement('span');toggle.className='router-run-toggle';toggle.textContent=open?'▾':'▸';header.append(toggle)}const dateEl=document.createElement('span');dateEl.className='router-run-date';dateEl.textContent=date;const timeEl=document.createElement('span');timeEl.className='router-run-time';timeEl.textContent=time;const prompt=document.createElement('span');prompt.className='router-run-prompt';prompt.textContent=first.prompt_preview||t('not.recoverable');prompt.title=prompt.textContent;const stateEl=document.createElement('span');stateEl.className=`router-run-state ${state==='running'?'running':''}`;stateEl.textContent=state==='running'?`▶ ${t('agents.pill.running')} · ${(parent?.children||[]).filter(child=>child.state==='running').length||1} ${t('agents.pill.agent')}`:t('state.done.short');const total=document.createElement('span');total.className='router-run-total';total.innerHTML=`<b>${t('total.routing.decisions')}</b>`;total.append(` ${accountingCalls.length}`);const routes=document.createElement('span');routes.className='router-run-routes';appendRoutePills(routes,accountingCalls);const workers=document.createElement('span');workers.className='router-run-workers';workers.textContent=`${workerCalls} ${t('run.worker.routing')}`;header.append(dateEl,timeEl,prompt,stateEl,total,routes,workers);if(hasDetails)header.addEventListener('click',()=>{setPromptExpanded(key,!open);render()});run.append(header);if(hasDetails&&open){const panel=document.createElement('section');panel.className='execution-tree-panel';panel.setAttribute('aria-label',t('exec.tree.label'));panel.append(executionTree(scope.nodes));run.append(panel)}runs.append(run)}};
let refreshInFlight=null;
function refreshDashboard(){if(refreshInFlight)return refreshInFlight;const button=$('refresh'),previousLabel=button.textContent;button.disabled=true;button.textContent=t('status.refresh.short');$('status').className='status refreshing';$('status').textContent=entries.length?t('status.refreshing')+t('status.kept'):t('status.loading');refreshInFlight=(async()=>{try{const [entriesResponse,agentsResponse]=await Promise.all([fetch(`/api/entries?roots=${$('last').value}`,{cache:'no-store'}),fetch('/api/agents',{cache:'no-store'})]);if(!entriesResponse.ok)throw new Error(`Entries HTTP ${entriesResponse.status}`);if(!agentsResponse.ok)throw new Error(`Agents HTTP ${agentsResponse.status}`);const [payload,activity]=await Promise.all([entriesResponse.json(),agentsResponse.json()]);entries=payload.entries||[];agentActivity=activity;render()}catch(error){$('status').className='status error';$('status').textContent=`${t('status.error.prefix')}${error.message}${t('status.error.kept')}`}finally{button.disabled=false;button.textContent=previousLabel;refreshInFlight=null}})();return refreshInFlight}
$('last').addEventListener('change',refreshDashboard);$('refresh').addEventListener('click',refreshDashboard);setInterval(()=>{if($('auto').checked)refreshDashboard()},3000);refreshDashboard();
</script><style>.agent-branch{margin-top:14px;padding:14px;border:1px solid #2c3951;border-radius:12px;background:#0b111c}.agent-parent-head{display:flex;justify-content:space-between;gap:12px;align-items:center}.agent-parent-prompt{margin-top:7px;color:#e9eef8;font-weight:650;white-space:normal;overflow-wrap:anywhere}.agent-children{position:relative;margin:13px 0 0 10px;padding-left:20px;border-left:1px solid #334158}.agent-child-row{display:grid;grid-template-columns:10px minmax(0,1fr) auto;gap:10px;align-items:center;position:relative;padding:10px 0}.agent-child-row:before{content:'';position:absolute;left:-20px;top:50%;width:18px;border-top:1px solid #334158}.agent-child-body{min-width:0}.agent-reason{font-weight:750;color:#e9eef8}.agent-child-goal{margin-top:2px;color:var(--muted);font-size:12px;white-space:normal;overflow-wrap:anywhere}.agent-meta{white-space:pre-line;min-width:120px}@media(max-width:700px){.agent-child-row{grid-template-columns:10px minmax(0,1fr)}.agent-child-row .agent-meta{grid-column:2;text-align:left}.agent-parent-head{align-items:flex-start;flex-direction:column;gap:2px}}.agent-main-card{margin-top:14px;border:1px solid #33435d;border-radius:16px;background:linear-gradient(145deg,rgba(20,29,45,.96),rgba(9,14,24,.98));overflow:hidden;box-shadow:0 12px 28px rgba(0,0,0,.16)}.agent-main-toggle{width:100%;display:grid;grid-template-columns:92px 64px minmax(0,1fr) auto;align-items:center;gap:10px;padding:15px 16px;border:0;border-radius:0;background:transparent;text-align:left}.agent-main-toggle:hover{background:rgba(167,139,250,.08);border-color:transparent}.agent-main-toggle:focus-visible{outline:2px solid var(--accent);outline-offset:-3px}.agent-main-heading{min-width:0;display:grid;gap:4px}.agent-main-date,.agent-main-time{color:#9dabbe;font-size:11px;font-variant-numeric:tabular-nums;white-space:nowrap}.agent-main-time{color:#c3cede}.agent-main-preview{color:#eef3ff;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.agent-main-started{color:#9dabbe;font-size:11px;font-variant-numeric:tabular-nums}.agent-main-stats{display:flex;align-items:center;justify-content:flex-end;gap:7px;flex-shrink:0}.agent-stat,.agent-worker-state,.agent-read-only{border:1px solid #394a66;border-radius:99px;padding:3px 8px;color:#aebbd0;font-size:11px;font-weight:750;white-space:nowrap}.agent-read-only{color:#ffd18a;border-color:rgba(255,180,84,.6);background:rgba(255,180,84,.1)}.agent-stat.running,.agent-worker-state.running{color:#b8f6a6;border-color:rgba(136,227,111,.55);background:rgba(136,227,111,.1)}.agent-chevron{color:#c4b5fd;font-size:18px;line-height:1}.agent-main-detail{padding:0 16px 16px;border-top:1px solid rgba(51,67,93,.72)}.agent-full-prompt{padding:13px 0;color:#c8d3e6;font-size:13px;line-height:1.55;white-space:normal;overflow-wrap:anywhere}.agent-worker-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:9px}.agent-worker-card{min-width:0;padding:11px 12px;border:1px solid #2a3850;border-radius:12px;background:linear-gradient(145deg,#101927,#0b111c)}.agent-worker-card.running{border-color:rgba(136,227,111,.46);box-shadow:inset 3px 0 0 #88e36f}.agent-worker-top{display:flex;align-items:center;justify-content:space-between;gap:8px}.agent-worker-purpose{color:#eef3ff;font-size:12px;font-weight:800}.agent-worker-details{margin-top:9px;color:#aebbd0;font-size:12px}.agent-worker-model{margin-top:3px;color:#8f9fb7;font:11px ui-monospace,SFMono-Regular,Consolas,monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}@media(max-width:700px){.agent-main-toggle{align-items:flex-start}.agent-main-preview{white-space:normal}.agent-main-stats{flex-wrap:wrap;max-width:105px}.agent-worker-grid{grid-template-columns:1fr}}.router-agent-workers{display:flex;overflow-x:auto;gap:8px;padding:2px 0 7px}.router-agent-workers .agent-worker-card{flex:0 0 300px;padding:8px 10px}.router-agent-workers .agent-worker-details{margin-top:5px}.router-agent-workers .agent-worker-model{margin-top:2px}.task-tree{padding:5px 0;display:grid;gap:2px}.task-tree-row{--tree-depth:0;display:grid;grid-template-columns:18px 92px minmax(180px,1fr) max-content;gap:9px;align-items:center;min-height:38px;padding:7px 10px 7px calc(10px + var(--tree-depth) * 28px);border-left:1px solid #334158;background:rgba(8,13,22,.45);font-size:12px}.task-tree-row:hover{background:#151d2b}.task-tree-row.running{border-left-color:#88e36f;box-shadow:inset 3px 0 0 rgba(136,227,111,.6)}.task-tree-branch,.task-tree-toggle{color:#8f9fb7;font:14px ui-monospace,SFMono-Regular,Consolas,monospace}.task-tree-toggle{width:18px;padding:0;border:0;background:transparent;text-align:center}.task-tree-toggle:hover{color:#c4b5fd;border-color:transparent}.task-tree-description{min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#e9eef8}.word-wrap .task-tree-description{white-space:normal;overflow:visible;text-overflow:clip;overflow-wrap:anywhere}.task-tree-state{border:1px solid #394a66;border-radius:99px;padding:3px 7px;color:#aebbd0;font-size:10px;font-weight:800}.task-tree-state.running{color:#b8f6a6;border-color:rgba(136,227,111,.55);background:rgba(136,227,111,.1)}@media(max-width:850px){.task-tree-row{grid-template-columns:18px 78px minmax(150px,1fr) max-content;gap:7px}.task-tree-state{grid-column:3;grid-row:2;justify-self:end}}@media(max-width:560px){.task-tree-row{grid-template-columns:18px 1fr;gap:4px;padding-left:calc(8px + var(--tree-depth) * 18px)}.task-tree-description,.task-tree-state{grid-column:2}.task-tree-state{grid-row:auto;justify-self:start}}.agent-worker-goal-details{min-width:0;max-width:100%;margin-top:8px}.agent-worker-goal-details summary{cursor:pointer;color:#c4b5fd;font-weight:700}.agent-worker-goal{margin-top:7px;max-width:100%;white-space:normal;overflow-wrap:anywhere;word-break:break-word}.router-runs{display:grid;gap:14px}.router-run{overflow:hidden;border:1px solid #304159;border-radius:15px;background:linear-gradient(145deg,rgba(13,22,34,.98),rgba(7,13,22,.98));box-shadow:0 12px 30px rgba(0,0,0,.16)}.router-run-header{width:100%;display:grid;grid-template-columns:24px 106px 78px minmax(180px,1fr) max-content max-content max-content max-content;align-items:center;gap:12px;padding:18px 22px;border:0;border-bottom:1px solid transparent;border-radius:0;background:transparent;text-align:left}.router-run-header.no-details{grid-template-columns:106px 78px minmax(180px,1fr) max-content max-content max-content max-content}.router-run.is-open .router-run-header{border-bottom-color:#304159}.router-run-header:hover{border-color:transparent;background:rgba(167,139,250,.055)}.router-run-header:focus-visible,.execution-tree-toggle:focus-visible{outline:2px solid var(--accent);outline-offset:-3px}.router-run-toggle,.execution-tree-toggle{color:#c4b5fd;font-size:27px;line-height:1}.router-run-date,.router-run-time{font:600 14px ui-monospace,SFMono-Regular,Consolas,monospace;color:#e9eef8;white-space:nowrap}.router-run-prompt{min-width:0;font-size:15px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.router-run-state,.execution-tree-state{border:1px solid #89a1be;border-radius:9px;padding:4px 9px;color:#c9d9ef;font:800 11px ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap}.router-run-state.running,.execution-tree-state.running{border-color:var(--terra);color:#a8f18d;background:rgba(136,227,111,.08)}.router-run-total{display:flex;gap:9px;color:#e6edf8;font:600 14px ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap}.router-run-total b{color:#91aaca;font-size:11px;letter-spacing:.04em}.router-run-routes,.execution-tree-routes{display:flex;align-items:center;gap:5px;min-width:0}.router-run-routes .pill,.execution-tree-routes .pill{margin:0}.router-run-workers{white-space:nowrap;color:#e9eef8;font-size:12px}.execution-tree-panel{margin:0;overflow:auto;background:transparent;border:0;border-radius:0}.execution-tree{position:relative;padding:0 22px 12px}.execution-tree-row{--tree-depth:0;display:grid;grid-template-columns:22px 16px 150px minmax(180px,1fr) max-content max-content max-content;align-items:center;gap:11px;min-height:52px;padding:7px 10px 7px calc(44px + var(--tree-depth) * 38px);position:relative;border-bottom:1px solid rgba(48,65,89,.48)}.execution-tree-row:last-child{border-bottom:0}.execution-tree-row:before{content:'';position:absolute;left:calc(34px + var(--tree-depth) * 38px);top:0;bottom:50%;border-left:1px solid #50637d}.execution-tree-row.task-tree-has-next-sibling:before{bottom:0}.execution-tree-row:after{content:'';position:absolute;left:calc(34px + var(--tree-depth) * 38px);top:50%;width:20px;border-top:1px solid #50637d}.execution-tree-children{position:relative}.execution-tree-children:before{content:'';position:absolute;left:calc(34px + var(--tree-depth,0) * 38px + 38px);top:0;bottom:26px;border-left:1px solid #50637d}.execution-tree-toggle,.execution-tree-spacer{position:relative;z-index:1;width:24px;min-height:28px;padding:0;border:0;background:transparent;text-align:center}.execution-tree-toggle{cursor:pointer}.execution-tree-spacer{color:transparent}.task-tree-marker{position:relative;z-index:1;width:12px;height:12px;background:#eff4fb;border-radius:50%;box-shadow:0 0 0 2px #0b1420}.task-tree-marker.lifecycle{background:#90ec70;box-shadow:0 0 12px rgba(144,236,112,.65)}.task-tree-marker.terra{background:#c49bff;box-shadow:0 0 10px rgba(196,155,255,.45)}.task-tree-marker.opus5{background:var(--opus5);box-shadow:0 0 10px rgba(214,149,255,.5)}.task-tree-marker.sonnet5{background:var(--sonnet5);box-shadow:0 0 10px rgba(159,180,255,.5)}.task-tree-opus5 .execution-tree-model{color:var(--opus5)}.execution-tree-model{font:800 14px ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap}.task-tree-lifecycle .execution-tree-model{color:#d9a9ff}.task-tree-terra .execution-tree-model{color:#a8f18d}.execution-tree-description{min-width:0;color:#eff3fa;font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.execution-tree-accounting{white-space:nowrap;color:#d3ddea;font-size:12px}.execution-tree-state{justify-self:end}.word-wrap .router-run-prompt,.word-wrap .execution-tree-description{white-space:normal;overflow:visible;text-overflow:clip;overflow-wrap:anywhere}.word-wrap .execution-tree-row{align-items:start;padding-top:13px;padding-bottom:13px}.word-wrap .execution-tree-row:after{top:24px}@media(max-width:1120px){.router-run-header{grid-template-columns:24px 100px 76px minmax(180px,1fr) max-content;gap:9px}.router-run-total{grid-column:5}.router-run-routes{grid-column:4 / -1;grid-row:2}.router-run-workers{grid-column:5;grid-row:2}.execution-tree-row{grid-template-columns:18px 16px 135px minmax(150px,1fr) max-content;gap:8px}.execution-tree-routes{grid-column:4 / -1}.execution-tree-state{grid-column:5;grid-row:2}}@media(max-width:700px){.router-run-header{grid-template-columns:22px 1fr max-content;padding:14px}.router-run-date{grid-column:2}.router-run-time{grid-column:3}.router-run-prompt{grid-column:2 / -1;grid-row:2}.router-run-state{grid-column:2;grid-row:3;justify-self:start}.router-run-total{grid-column:3;grid-row:3}.router-run-routes{grid-column:2 / -1;grid-row:4}.router-run-workers{grid-column:2;grid-row:5}.execution-tree-panel{margin:10px}.execution-tree{padding:8px}.execution-tree-row{grid-template-columns:18px 16px minmax(0,1fr) max-content;padding-left:calc(8px + var(--tree-depth) * 24px);gap:7px}.execution-tree-row:before{left:calc(14px + var(--tree-depth) * 24px)}.execution-tree-row:after{left:calc(14px + var(--tree-depth) * 24px)}.execution-tree-model{grid-column:3}.execution-tree-description{grid-column:3 / -1;grid-row:2}.execution-tree-accounting{grid-column:3;grid-row:3}.execution-tree-routes{grid-column:3 / -1;grid-row:4}.execution-tree-state{grid-column:4;grid-row:3;justify-self:end}}</style></body></html>'''


SYNTHETIC_LIFECYCLE_PREFIXES = (
    "[ASYNC DELEGATION BATCH COMPLETE",
    "[ASYNC DELEGATION COMPLETE",
    "[Your active task list was preserved across context compression]",
    "[IMPORTANT: Background process ",
    "Review the conversation above and consider saving to memory if appropriate.",
    "[CONTEXT COMPACTION",
)


def _session_id(entry: dict) -> str:
    return str(entry.get("turn_id") or "").split(":", 1)[0]


def _is_lifecycle(entry: dict) -> bool:
    prompt = str(entry.get("prompt_preview") or "")
    turn_id = str(entry.get("turn_id") or "")
    return bool(entry.get("is_internal_prompt")) or ":sa-" in turn_id or any(
        prompt.startswith(prefix) for prefix in SYNTHETIC_LIFECYCLE_PREFIXES
    )


def _activity_children(parent: dict) -> list[dict]:
    return list(parent.get("children") or [])


def _descendant_sessions(parent: dict, parents_by_session: dict[str, dict]) -> set[str]:
    found: set[str] = set()
    pending = _activity_children(parent)
    while pending:
        child = pending.pop()
        session_id = str(child.get("agent_session_id") or "")
        if not session_id or session_id in found:
            continue
        found.add(session_id)
        nested = parents_by_session.get(session_id)
        if nested:
            pending.extend(_activity_children(nested))
        pending.extend(_activity_children(child))
    return found


def select_recent_root_closure(entries: list[dict], activity: dict, root_limit: int) -> tuple[list[dict], int]:
    """Return the latest visible roots and only their root/child/lifecycle records."""
    parents = list(activity.get("parents") or [])
    parents_by_session = {str(parent.get("session_id") or ""): parent for parent in parents}
    all_child_sessions = {
        session_id
        for parent in parents
        for session_id in _descendant_sessions(parent, parents_by_session)
    }

    groups: dict[tuple[str, str], list[dict]] = {}
    for entry in entries:
        if _is_lifecycle(entry) or _session_id(entry) in all_child_sessions:
            continue
        key = (_session_id(entry), str(entry.get("prompt_preview") or f"__missing__:{entry.get('turn_id', '')}"))
        groups.setdefault(key, []).append(entry)
    selected_groups = list(groups.values())[-root_limit:]
    if not selected_groups:
        return [], 0

    selected_ids = {id(entry) for group in selected_groups for entry in group}
    selected_turns = {str(entry.get("turn_id") or "") for group in selected_groups for entry in group}
    selected_sessions = {_session_id(group[0]) for group in selected_groups}
    selected_child_sessions: set[str] = set()
    for group in selected_groups:
        session_id = _session_id(group[0])
        prompt = str(group[0].get("prompt_preview") or "").strip()
        candidates = [parent for parent in parents if str(parent.get("session_id") or "") == session_id]
        parent = next((item for item in candidates if str(item.get("prompt") or "").strip() == prompt), None)
        if parent is None and len(candidates) == 1:
            parent = candidates[0]
        if parent:
            selected_child_sessions.update(_descendant_sessions(parent, parents_by_session))

    for entry in entries:
        if _session_id(entry) in selected_child_sessions:
            selected_ids.add(id(entry))
            continue
        if not _is_lifecycle(entry):
            continue
        parent_turn = str(entry.get("parent_turn_id") or "")
        if parent_turn in selected_turns or (not parent_turn and _session_id(entry) in selected_sessions):
            selected_ids.add(id(entry))
    return [entry for entry in entries if id(entry) in selected_ids], len(selected_groups)


class Handler(BaseHTTPRequestHandler):
    log_path = DEFAULT_LOG
    state_db_path = DEFAULT_STATE_DB
    agent_log_path = DEFAULT_AGENT_LOG
    bridge_lifecycle_path = DEFAULT_BRIDGE_LIFECYCLE

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/config":
            if yaml is None:
                self._send(500, json.dumps({"error": "yaml not available"}).encode("utf-8"), "application/json")
                return
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                if "callable" in data:
                    config["callable"] = data["callable"]
                if "default_model" in data:
                    requested_default = str(data["default_model"])
                    callable_tiers = config.get("callable") or {}
                    fallbacks = config.get("fallbacks") or {}
                    candidate = requested_default
                    visited = {candidate}
                    while not callable_tiers.get(candidate, True):
                        candidate = str(fallbacks.get(candidate) or "")
                        if not candidate or candidate in visited:
                            self._send(
                                400,
                                json.dumps({
                                    "error": f"No enabled fallback for default model '{requested_default}'",
                                    "success": False,
                                }).encode("utf-8"),
                                "application/json",
                            )
                            return
                        visited.add(candidate)
                    config["default_model"] = candidate
                    data["default_model"] = candidate
                    # Szinkronizáljuk a Hermes config orchestrator részét is
                    hermes_config_path = Path.home() / ".hermes" / "config.yaml"
                    if hermes_config_path.exists() and yaml:
                        try:
                            with open(hermes_config_path, "r", encoding="utf-8") as f:
                                hermes_config = yaml.safe_load(f) or {}
                            default_model = data["default_model"]
                            model_name = config.get("models", {}).get(default_model, "")
                            tier_providers = config.get("tier_providers", {})
                            provider = tier_providers.get(default_model, "openai-codex")

                            # Frissítjük a globális model konfigurációt (orchestrator).
                            # A delegation külön child-runtime; orchestratorváltáskor nem
                            # írjuk felül, így Qwen parent mellett Terra maradhat a child.
                            if "model" not in hermes_config:
                                hermes_config["model"] = {}
                            hermes_config["model"]["default"] = model_name
                            hermes_config["model"]["provider"] = provider

                            provider_config = hermes_config.get("providers", {}).get(provider, {})
                            transport = provider_config.get("transport", "")
                            if transport == "anthropic_messages":
                                hermes_config["model"]["api_mode"] = "anthropic_messages"
                                base_url = provider_config.get("base_url") or provider_config.get("api")
                                if base_url:
                                    hermes_config["model"]["base_url"] = base_url
                                if provider_config.get("api_key"):
                                    hermes_config["model"]["api_key"] = provider_config["api_key"]
                            elif provider == "openai-codex":
                                hermes_config["model"]["api_mode"] = "codex_responses"
                                hermes_config["model"].pop("base_url", None)
                                hermes_config["model"].pop("api_key", None)
                            else:
                                hermes_config["model"]["api_mode"] = "chat_completions"
                                base_url = provider_config.get("base_url") or provider_config.get("api")
                                if base_url:
                                    hermes_config["model"]["base_url"] = base_url
                                if provider_config.get("api_key"):
                                    hermes_config["model"]["api_key"] = provider_config["api_key"]

                            delegation_config = hermes_config.setdefault("delegation", {})
                            if "targets" not in delegation_config:
                                delegation_config["targets"] = {}
                            delegation_config["targets"][default_model] = {
                                "provider": provider,
                                "model": model_name
                            }

                            with open(hermes_config_path, "w", encoding="utf-8") as f:
                                yaml.dump(hermes_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                        except Exception as e:
                            pass  # Nem kritikus hiba, folytatjuk
                with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                self._send(200, json.dumps({"success": True}).encode("utf-8"), "application/json")
            except Exception as e:
                self._send(500, json.dumps({"error": str(e), "success": False}).encode("utf-8"), "application/json")
            return
        self._send(404, b'{"error":"not found"}', "application/json")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/agents":
            body = json.dumps(
                load_agent_activity(
                    self.state_db_path,
                    log_path=self.agent_log_path,
                    router_log_path=self.log_path,
                    bridge_lifecycle_path=self.bridge_lifecycle_path,
                ),
                ensure_ascii=False,
            ).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        if parsed.path == "/api/entries":
            try:
                requested_root_limit = int(
                    parse_qs(parsed.query).get("roots", [str(DEFAULT_ROOT_LIMIT)])[0]
                )
            except ValueError:
                requested_root_limit = DEFAULT_ROOT_LIMIT
            if requested_root_limit not in {1, 5, 10}:
                requested_root_limit = DEFAULT_ROOT_LIMIT
            source_entries = load_entries(self.log_path)[-RAW_HISTORY_LIMIT:]
            activity = load_agent_activity(
                self.state_db_path,
                log_path=self.agent_log_path,
                router_log_path=self.log_path,
                bridge_lifecycle_path=self.bridge_lifecycle_path,
            )
            entries, selected_root_count = select_recent_root_closure(
                source_entries, activity, requested_root_limit
            )
            body = json.dumps(
                {
                    "entries": entries,
                    "requested_root_limit": requested_root_limit,
                    "selected_root_count": selected_root_count,
                    "source_entry_count": len(source_entries),
                    "raw_history_limit": RAW_HISTORY_LIMIT,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        if parsed.path == "/api/config":
            if yaml is None:
                self._send(500, json.dumps({"error": "yaml not available"}).encode("utf-8"), "application/json")
                return
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                response = {
                    "callable": config.get("callable", {}),
                    "default_model": config.get("default_model", "terra"),
                    **_router_status(),
                }
                self._send(200, json.dumps(response).encode("utf-8"), "application/json")
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}).encode("utf-8"), "application/json")
            return
        self._send(404, b'{"error":"not found"}', "application/json")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="listen address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="listen port (default: 8765)")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="JSONL log path")
    args = parser.parse_args()
    Handler.log_path = args.log.expanduser()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Model-router log viewer: http://{args.host}:{args.port}", flush=True)
    print(f"Log: {Handler.log_path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
