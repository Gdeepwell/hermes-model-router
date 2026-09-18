import json
import os
import tempfile
import threading
import time
import subprocess
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from unittest.mock import patch

import web_viewer
from web_viewer import HTML, Handler


class DashboardProbeMixin:
    """Helpers that slice the shipped dashboard source so probes stay honest.

    Separate from the test classes so a suite can reuse them without inheriting —
    and re-running — every test the other class declares.
    """

    def execution_source(self):
        start = HTML.index("/* Reference execution-tree renderer:")
        end = HTML.index("function executionState", start)
        return HTML[start:end]

    def i18n(self, key):
        """Return the (english, hungarian) pair declared for one i18n key.

        Structural assertions anchor on the key; this checks the copy itself,
        so a translation change can never silently break an unrelated test.
        """
        import re

        english = HTML[HTML.index("  en: {"):HTML.index("  hu: {")]
        hungarian = HTML[HTML.index("  hu: {"):HTML.index("\n};", HTML.index("  hu: {"))]
        pattern = r"^\s*'%s':\s*'((?:[^'\\]|\\.)*)'" % re.escape(key)
        found = []
        for block, language in ((english, "en"), (hungarian, "hu")):
            match = re.search(pattern, block, re.M)
            self.assertIsNotNone(match, f"i18n key {key!r} missing from the {language} dictionary")
            found.append(match.group(1))
        return tuple(found)

    def i18n_runtime(self, language="en"):
        """The dashboard's real I18N dictionary and t(), ready to run in node.

        The execution-tree renderer calls t(), so a probe without it dies with
        a ReferenceError. Slicing the live source keeps the probes honest —
        a renamed key fails here rather than silently falling back.
        """
        start = HTML.index("const I18N = {")
        end = HTML.index("function applyLanguage()")
        runtime = HTML[start:end].replace("localStorage.getItem('model-router-lang')", "null")
        return runtime.replace("let currentLang = null || 'en';", f"let currentLang = {language!r};")

    def javascript_function(self, name):
        start = HTML.index(f"function {name}")
        brace = HTML.index("{", start)
        depth = 0
        for index in range(brace, len(HTML)):
            if HTML[index] == "{":
                depth += 1
            elif HTML[index] == "}":
                depth -= 1
                if depth == 0:
                    return HTML[start:index + 1]
        self.fail(f"unterminated JavaScript function: {name}")

class ModelRouterDashboardTests(DashboardProbeMixin, unittest.TestCase):
    def test_execution_tree_counts_are_labeled_as_routing_decisions(self):
        self.assertIn('<div class="cards"><div class="card"><div class="n" id="total">0</div>', HTML)
        self.assertIn('<div class="k" data-i18n="card.total">', HTML)
        self.assertNotIn('.cards{display:none}', HTML)
        renderer = HTML[HTML.rindex("render=function(){"):]
        self.assertIn("total.innerHTML=`<b>${t('total.routing.decisions')}</b>`", renderer)
        self.assertIn("workers.textContent=`${workerCalls} ${t('run.worker.routing')}`", renderer)
        self.assertEqual(self.i18n('total.routing.decisions'), ('TOTAL ROUTING DECISIONS', 'ÖSSZES ROUTING DÖNTÉS'))
        self.assertNotIn('ÖSSZES HÍVÁS', renderer)
        self.assertNotIn('worker-hívás', renderer)

    def test_the_two_claude_tiers_are_styled_the_same_way(self):
        """Duplicating a rule for a new tier is easy to get wrong: dropping the
        selector prefix turns `.task-tree-marker.opus5{background:…}` into a bare
        `.sonnet5{background:…}`, which then paints the whole summary card in the
        tier colour instead of a four-pixel marker."""
        import re

        css = "".join(re.findall(r"<style>(.*?)</style>", HTML, re.S))
        selectors = {
            tier: sorted(
                match.group(1).replace(tier, "<tier>")
                for match in re.finditer(r"([^{};]*\.%s[^{};]*)\{" % tier, css)
            )
            for tier in ("opus5", "sonnet5")
        }
        self.assertTrue(selectors["opus5"], "expected opus5 to carry tier styling")
        self.assertEqual(selectors["opus5"], selectors["sonnet5"])

    def test_final_router_renderer_refreshes_each_summary_counter(self):
        """Asserted against the cards themselves rather than a copied literal:
        adding a tier used to mean editing four separate lists, and a counter
        left out of one of them renders as a card frozen at zero."""
        import re

        renderer = HTML[HTML.rindex("render=function(){"):]
        card_ids = re.findall(r'<div class="n" id="([a-z0-9]+)">', HTML)
        loop = re.search(r"for\(const id of \[([^\]]+)\]\)\$\(id\)\.textContent=summary\[id\]", renderer)
        self.assertIsNotNone(loop)
        refreshed = [name.strip("'") for name in loop.group(1).split(",")]
        # 'total' has its own line; every other card must be in the loop or it
        # renders frozen at zero, which is what the Qwen card did.
        self.assertEqual(sorted(refreshed), sorted(set(card_ids) - {"total"}))
        self.assertIn("$('total').textContent=summary.total", renderer)

    def test_recent_selector_is_a_root_prompt_limit(self):
        select_start = HTML.index('<select id="last">')
        label_start = HTML.rfind('<label>', 0, select_start)
        selector = HTML[label_start:HTML.index('</label>', select_start)]
        self.assertEqual(
            selector,
            '<label><span data-i18n="router.last.label">Utolsó root promptok</span>'
            '<select id="last"><option>1</option><option>5</option><option selected>10</option></select>',
        )
        self.assertEqual(self.i18n('router.last.label'), ('Last root prompts', 'Utolsó root promptok'))
        self.assertIn("fetch(`/api/entries?roots=${$('last').value}`", HTML)
        self.assertIn(
            "`${limitedRoots.length} ${t('status.rootprompts')} · "
            "${new Date().toLocaleTimeString(t('status.locale'))}`",
            HTML,
        )

    def test_root_limit_keeps_latest_roots_and_all_of_their_raw_records(self):
        source = "\n".join(self.javascript_function(name) for name in (
            "limitRootRuns", "rawEntryKey", "rawEntriesForRootRuns"
        ))
        roots = [
            {"id": "old", "rawEntries": [{"id": "old-root"}] + [{"id": f"old-child-{index}"} for index in range(230)]},
            {"id": "middle", "rawEntries": [{"id": "middle-root"}]},
            {"id": "latest", "rawEntries": [{"id": "latest-root"}, {"id": "latest-lifecycle"}]},
        ]
        all_entries = roots[0]["rawEntries"] + roots[1]["rawEntries"] + roots[2]["rawEntries"]
        probe = (
            source + "\nconst roots=" + json.dumps(roots) + ";"
            "const allEntries=" + json.dumps(all_entries) + ";"
            "const limited=limitRootRuns(roots,2);"
            "console.log(JSON.stringify({roots:limited.map(run=>run.id),raw:rawEntriesForRootRuns(limited,allEntries).map(entry=>entry.id)}));"
        )
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        self.assertEqual(json.loads(result.stdout), {
            "roots": ["middle", "latest"],
            "raw": ["middle-root", "latest-root", "latest-lifecycle"],
        })

    def test_more_than_200_newer_lifecycle_records_do_not_hide_the_root_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "router.jsonl"
            records = [{
                "timestamp": "2026-08-03T00:00:00+00:00",
                "turn_id": "root-session:root-turn",
                "prompt_preview": "Visible root",
                "tier": "sol",
            }]
            records.extend({
                "timestamp": f"2026-08-03T00:{index // 60:02d}:{index % 60:02d}+00:00",
                "turn_id": f"root-session:sa-{index}",
                "parent_turn_id": "root-session:root-turn",
                "prompt_preview": f"[ASYNC DELEGATION COMPLETE — lifecycle-{index}]",
                "is_internal_prompt": True,
                "tier": "sol",
            } for index in range(1, 251))
            log_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
            original_path = Handler.log_path
            Handler.log_path = log_path
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{server.server_port}/api/entries?roots=5"
                ) as response:
                    payload = json.load(response)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
                Handler.log_path = original_path

            source = "let agentActivity={parents:[]};const syntheticLifecyclePrefixes=[];\n" + "\n".join(
                self.javascript_function(name) for name in (
                    "childSessionIds",
                    "turnBelongsToChild",
                    "isSyntheticLifecyclePrompt",
                    "visibleEntries",
                    "sessionIdFromTurn",
                    "systemEntriesForRoot",
                    "promptKey",
                    "promptGroups",
                    "limitRootRuns",
                )
            )
            probe = (
                source + "\nconst entries=" + json.dumps(payload["entries"]) + ";"
                "const roots=limitRootRuns(promptGroups(visibleEntries(entries)),5);"
                "console.log(JSON.stringify({prompts:roots.map(group=>group[0].prompt_preview),"
                "lifecycle:systemEntriesForRoot(roots[0][0],entries).length}));"
            )
            result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)

            self.assertEqual(len(payload["entries"]), 251)
            self.assertEqual(payload["requested_root_limit"], 5)
            self.assertEqual(payload["raw_history_limit"], 10000)
            self.assertEqual(json.loads(result.stdout), {
                "prompts": ["Visible root"],
                "lifecycle": 250,
            })

    def test_only_top_summary_cards_use_expanded_model_family_labels(self):
        cards = HTML[HTML.index('<div class="cards">'):HTML.index('<div id="runs"')]
        for key, label in [('card.luna', 'GPT-5.6 Luna'), ('card.spark', 'GPT-5.3 Spark'),
                           ('card.terra', 'GPT-5.6 Terra'), ('card.sol', 'GPT-5.6 Sol'),
                           ('card.opus5', 'Claude Opus 5')]:
            self.assertIn(f'<div class="k" data-i18n="{key}">{label}</div>', cards)
            self.assertEqual(self.i18n(key), (label, label))
        # Since Task 7 the static #tier select only carries the "all" option;
        # the per-tier <option>s come from tierFilterOptions(), grouped by
        # account (see AccountGroupTests).
        self.assertIn("return effort?`${tier} · ${effort}`:tier", HTML)
        self.assertIn("String(raw?.tier||raw?.model||node.model||kind).toUpperCase()", HTML)
        router_run_markup = HTML[HTML.index('<div id="runs"'):HTML.index('<section id="settings-panel"')]
        self.assertNotIn('GPT-5.6 Sol', router_run_markup)
        self.assertIn('<div class="card sol">', cards)
        self.assertIn('.pill.sol{color:var(--sol)}', HTML)

    def test_workers_remain_inside_router_prompt_rows(self):
        self.assertNotIn('data-tab="agents"', HTML)
        self.assertIn("function parentForGroup(group)", HTML)
        self.assertIn("function childEntries(child,list,seen=new Set())", HTML)

    def test_root_prompt_disclosure_is_independent_of_word_wrap(self):
        self.assertIn(
            '<label class="check"><input id="word-wrap" type="checkbox"> '
            '<span data-i18n="router.wordwrap">Sortörés</span></label>',
            HTML,
        )
        self.assertIn("hasDetails=scope.nodes.length>0", HTML)
        self.assertIn("if(hasDetails&&open)", HTML)
        self.assertIn(".word-wrap .router-run-prompt,.word-wrap .execution-tree-description", HTML)
        self.assertNotIn("detailsEnabled=$('word-wrap').checked", HTML)

    def test_empty_main_run_has_no_disclosure_or_detail_panel(self):
        renderer = HTML[HTML.rindex("render=function(){"):]
        self.assertIn("const header=document.createElement(hasDetails?'button':'div')", renderer)
        self.assertIn("router-run-header ${hasDetails?'':'no-details'}", renderer)
        self.assertIn("if(hasDetails){const toggle=document.createElement('span')", renderer)
        self.assertIn("if(hasDetails&&open)", renderer)

    def test_expanded_tree_has_no_inner_rounded_or_dark_panel_border(self):
        self.assertIn(".execution-tree-panel{margin:0;overflow:auto;background:transparent;border:0;border-radius:0}", HTML)
        self.assertNotIn(".execution-tree-panel{margin:16px 22px 22px;border:1px", HTML)

    def test_root_origin_and_nested_child_connector_contract(self):
        self.assertIn(".execution-tree-row:before{content:'';position:absolute;left:calc(34px + var(--tree-depth) * 38px);top:0;bottom:50%;border-left:1px solid #50637d}", HTML)
        self.assertIn(".execution-tree-row.task-tree-has-next-sibling:before{bottom:0}", HTML)
        self.assertIn(".execution-tree-children:before", HTML)
        self.assertIn("childrenEl.style.setProperty('--tree-depth',depth)", self.execution_source())
        self.assertIn("tree.append(childrenEl)", self.execution_source())

    def test_only_real_internal_node_has_one_clickable_disclosure(self):
        source = self.execution_source()
        self.assertIn("const disclosure=document.createElement(hasChildren?'button':'span')", source)
        self.assertIn("disclosure.className=hasChildren?'execution-tree-toggle':'execution-tree-spacer'", source)
        self.assertIn("disclosure.textContent=hasChildren?(open?'▾':'▸'):''", source)
        self.assertIn("event.stopPropagation();setTaskTreeOpen(node.id,!open);render()", source)
        self.assertNotIn("border-top:18px solid #c49bff", HTML)
        self.assertIn(".task-tree-marker.terra{background:#c49bff", HTML)

    def test_internal_disclosure_matches_root_and_leaf_has_no_toggle(self):
        source = self.execution_source()
        self.assertIn(".router-run-toggle,.execution-tree-toggle{color:#c4b5fd;font-size:27px;line-height:1}", HTML)
        self.assertIn(".execution-tree-toggle,.execution-tree-spacer{position:relative;z-index:1;width:24px;min-height:28px", HTML)
        self.assertIn("const disclosure=document.createElement(hasChildren?'button':'span')", source)
        self.assertIn("hasChildren?'execution-tree-toggle':'execution-tree-spacer'", source)

    def test_lifecycle_description_uses_own_prompt_or_short_safe_fallback(self):
        source = self.execution_source()
        self.assertIn("function lifecycleDescription(entry)", source)
        self.assertIn("return t('internal.router.step')", source)
        self.assertIn("task_description:lifecycleDescription(entry)", source)
        probe = self.i18n_runtime("hu") + source + "\nconst own={lifecycle_prompt:'Saját tárolt lifecycle feladat',prompt_preview:'[ASYNC DELEGATION COMPLETE — dump]'};const empty={prompt_preview:'[ASYNC DELEGATION COMPLETE — dump]'};console.log(JSON.stringify([lifecycleDescription(own),lifecycleDescription(empty)]));"
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        self.assertEqual(json.loads(result.stdout), ["Saját tárolt lifecycle feladat", "Belső router-lépés"])

    def test_lifecycle_description_prefers_human_completion_provenance(self):
        source = self.execution_source()
        probe = self.i18n_runtime() + source + "\nconst completion={event_kind:'async_delegation_completion',lifecycle_prompt:'Delegált feladat befejezési eseménye · Rövid redaktált feladat'};console.log(lifecycleDescription(completion));"
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        self.assertEqual(result.stdout.strip(), "Delegált feladat befejezési eseménye · Rövid redaktált feladat")

    def test_recursive_raw_accounting_makes_terra_include_spark_96(self):
        source = self.execution_source()
        self.assertIn("function executionCalls(node){return [...executionOwnCalls(node),...(node.children||[]).flatMap(executionCalls)]}", source)
        self.assertIn("const ownCalls=executionOwnCount(node),totalCalls=executionTotalCalls(node)", source)
        self.assertIn("appendRoutePills(routes,executionCalls(node))", source)
        self.assertIn("function executionScope(group,systemCalls,parent,tier='')", source)
        self.assertIn("accountingCalls=scope.calls", HTML)
        fixture = {"routed_calls": [{"tier": "terra", "effort": "medium"}] * 32,
                   "children": [{"routed_calls": [{"tier": "spark", "effort": "medium"}] * 48, "children": []},
                                {"routed_calls": [{"tier": "spark", "effort": "medium"}] * 48, "children": []}]}
        probe = source + "\nconst fixture=" + json.dumps(fixture) + ";console.log(JSON.stringify({total:executionTotalCalls(fixture),routes:executionCalls(fixture).map(executionRouteKey)}));"
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        observed = json.loads(result.stdout)
        self.assertEqual(observed["total"], 128)
        self.assertEqual(observed["routes"].count("terra · medium"), 32)
        self.assertEqual(observed["routes"].count("spark · medium"), 96)

    def test_scope_model_filter_reaches_nested_workers_and_prunes_other_calls(self):
        source = self.execution_source()
        group = [{"tier": "terra", "effort": "medium"}] * 4
        system = [{"tier": "sol", "effort": "medium"}] * 2
        parent = {"session_id": "parent", "children": [{
            "id": "supervisor", "routed_calls": [{"tier": "sol", "effort": "medium"}] * 29,
            "children": [
                {"id": "sol-leaf", "routed_calls": [{"tier": "sol", "effort": "medium"}] * 10, "children": []},
                {"id": "mixed-leaf", "routed_calls": ([{"tier": "spark", "effort": "medium"}] * 5 +
                                                          [{"tier": "terra", "effort": "medium"}] * 5), "children": []},
            ],
        }]}
        probe = self.i18n_runtime() + source + "\nconst scope=executionScope(" + json.dumps(group) + "," + json.dumps(system) + "," + json.dumps(parent) + ",'sol');const workerCalls=scope.nodes.filter(node=>node.kind!=='LIFECYCLE').flatMap(executionCalls);console.log(JSON.stringify({calls:scope.calls.map(executionRouteKey),workerCalls:workerCalls.length,tree:scope.nodes}));"
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        observed = json.loads(result.stdout)
        self.assertEqual(len(observed["calls"]), 41)
        self.assertEqual(observed["workerCalls"], 39)
        self.assertEqual(set(observed["calls"]), {"sol · medium"})
        self.assertEqual(len(observed["tree"]), 3)
        stack = list(observed["tree"])
        routed_tiers = set()
        while stack:
            node = stack.pop()
            routed_tiers.update(call["tier"] for call in node.get("routed_calls", []))
            stack.extend(node.get("children", []))
        self.assertEqual(routed_tiers, {"sol"})

    def test_summary_uses_unique_calls_represented_by_visible_request_scopes(self):
        source = self.execution_source()
        runs = [
            ([{"tier": "terra"}] * 27 + [{"tier": "sol"}] * 3 + [{"tier": "spark"}] * 5),
            ([{"tier": "terra"}] * 9 + [{"tier": "sol"}] * 31 + [{"tier": "spark"}] * 5),
            [{"tier": "terra"}],
            [{"tier": "terra"}],
        ]
        probe = source + "\nconst summary=executionSummary(" + json.dumps(runs) + ");console.log(JSON.stringify(summary));"
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        self.assertEqual(
            json.loads(result.stdout),
            {"total": 82, "luna": 0, "spark": 10, "terra": 38, "sol": 34,
             "opus5": 0, "sonnet5": 0, "haiku": 0, "qwen": 0},
        )
        renderer = HTML[HTML.rindex("render=function(){"):]
        self.assertIn("const summary=executionSummary(runData.map(run=>run.scope.calls))", renderer)
        self.assertNotIn("$('total').textContent=allEntries.length", renderer)

    def test_spark_effort_pill_and_raw_route_precedence(self):
        source = self.execution_source()
        self.assertIn("const tier=String(call?.tier||call?.model||'?').toLowerCase(),effort=String(call?.effort||'').toLowerCase()", source)
        self.assertIn("return effort?`${tier} · ${effort}`:tier", source)
        self.assertIn("function executionOwnCalls(node){const raw=Array.isArray(node?.routed_calls)?node.routed_calls:[];return raw.length?raw:[]}", source)
        self.assertIn("String(raw?.tier||raw?.model||node.model||kind).toUpperCase()", source)

    def test_opus5_is_enumerated_colored_and_filterable_in_grouped_and_raw_views(self):
        self.assertIn("--opus5:#d695ff", HTML)
        # Since Task 7 the static #tier select only carries the "all" option;
        # the per-tier <option>s come from tierFilterOptions() (see
        # AccountGroupTests.test_tier_filter_options_are_grouped_by_account_label).
        self.assertIn('class="card opus5"', HTML)
        self.assertIn(".pill.opus5{color:var(--opus5)}", HTML)
        self.assertIn(".task-tree-marker.opus5{background:var(--opus5)", HTML)
        source = self.execution_source()
        self.assertIn("source.includes('opus5')||source.includes('claude-opus-5')", source)
        self.assertIn("opus5:0", source)
        parent = {"children": [{"id": "external", "model": "claude-opus-5", "routed_calls": [{"tier": "opus5", "model": "claude-opus-5", "effort": "external"}], "children": []}]}
        probe = source + "\nfunction sessionIdFromTurn(entry){return String(entry?.turn_id||'').split(':')[0]}\nconst scope=executionScope([{tier:'terra'}],[]," + json.dumps(parent) + ",'opus5');console.log(JSON.stringify({calls:scope.calls,kinds:scope.nodes.map(executionKind),summary:executionSummary([scope.calls])}));"
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        observed = json.loads(result.stdout)
        self.assertEqual(observed["calls"][0]["tier"], "opus5")
        self.assertEqual(observed["kinds"], ["opus5"])
        self.assertEqual(observed["summary"]["opus5"], 1)

    def test_only_exact_prompt_group_in_same_session_inherits_running_parent(self):
        source = "\n".join(self.javascript_function(name) for name in (
            "sessionIdFromTurn", "promptsMatch", "parentForGroup", "executionState"
        ))
        # executionState returns a sentinel, not a label: its value used to be
        # the rendered Hungarian text, so translating the UI broke the branch.
        activity = {"parents": [{
            "session_id": "shared-session",
            "prompt": "Current live prompt",
            "children": [{"state": "running"}],
        }], "active_turns": []}
        old_group = [{"turn_id": "shared-session:old-turn", "prompt_preview": "Historical prompt"}]
        live_group = [{"turn_id": "shared-session:live-turn", "prompt_preview": "Current live prompt"}]
        probe = (
            "const agentActivity=" + json.dumps(activity) + ";\n" + source +
            "\nconst groups=" + json.dumps([old_group, live_group]) + ";" +
            "console.log(JSON.stringify(groups.map(group=>executionState(group,parentForGroup(group)))));"
        )
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        self.assertEqual(json.loads(result.stdout), ["done", "running"])

    def test_last_sibling_connector_does_not_continue_below_leaf(self):
        self.assertIn(".execution-tree-row:before{content:'';position:absolute;left:calc(34px + var(--tree-depth) * 38px);top:0;bottom:50%;border-left:1px solid #50637d}", HTML)
        self.assertIn(".execution-tree-row.task-tree-has-next-sibling:before{bottom:0}", HTML)

    def test_ten_roots_return_a_narrow_server_side_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "router.jsonl"
            records = []
            for root_index in range(30):
                root_turn = f"session-{root_index}:turn"
                records.append({"timestamp": f"2026-08-03T00:{root_index:02d}:00+00:00", "turn_id": root_turn,
                                "prompt_preview": f"root-{root_index}", "tier": "sol"})
                records.extend({"timestamp": f"2026-08-03T00:{root_index:02d}:{child:02d}+00:00",
                                "turn_id": f"session-{root_index}:sa-{child}", "parent_turn_id": root_turn,
                                "prompt_preview": f"[ASYNC DELEGATION COMPLETE {child}]", "is_internal_prompt": True,
                                "tier": "sol"} for child in range(1, 6))
            log_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
            original_path = Handler.log_path
            Handler.log_path = log_path
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/entries?roots=10") as response:
                    payload = json.load(response)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=2); Handler.log_path = original_path
        roots = [entry for entry in payload["entries"] if not entry.get("is_internal_prompt")]
        self.assertEqual([entry["prompt_preview"] for entry in roots], [f"root-{i}" for i in range(20, 30)])
        self.assertEqual(len(payload["entries"]), 60)
        self.assertEqual(payload["selected_root_count"], 10)
        self.assertEqual(payload["source_entry_count"], 180)

    def test_refresh_cycle_fetches_entries_and_agents_once_and_renders_once(self):
        self.assertIn("Promise.all([fetch(`/api/entries?roots=${$('last').value}`", HTML)
        self.assertIn("fetch('/api/agents'", HTML)
        cycle = self.javascript_function("refreshDashboard")
        self.assertEqual(cycle.count("render()"), 1)
        self.assertIn("agentActivity=activity", cycle)
        self.assertNotIn("renderAgents(", cycle)
        self.assertNotIn("async function load()", HTML)
        self.assertNotIn("async function loadAgents()", HTML)

    def test_polling_has_backpressure_and_keeps_old_rows_visible_while_refreshing(self):
        cycle = self.javascript_function("refreshDashboard")
        self.assertIn("if(refreshInFlight)return refreshInFlight", cycle)
        self.assertIn("finally", cycle)
        self.assertIn("refreshInFlight=null", cycle)
        self.assertNotIn("replaceChildren", cycle)
        self.assertIn("button.disabled=true", cycle)
        self.assertIn("button.textContent=t('status.refresh.short')", cycle)
        self.assertEqual(self.i18n('status.refresh.short'), ('Refresh…', 'Frissítés…'))
        self.assertIn("setInterval(()=>{if($('auto').checked)refreshDashboard()},3000)", HTML)

    def test_external_opus_child_card_has_badges_metrics_states_and_stable_run_disclosure(self):
        source = self.execution_source()
        self.assertIn("node.external", source)
        self.assertIn("badges.push(t('exec.external'))", source)
        self.assertIn("badges.push(t('exec.read.only'))", source)
        self.assertIn("badges.push(t('agents.requested.ro'))", source)
        self.assertEqual(self.i18n('exec.external'), ('EXTERNAL', 'KÜLSŐ'))
        self.assertEqual(self.i18n('exec.read.only'), ('READ-ONLY', 'READ-ONLY'))
        self.assertEqual(self.i18n('agents.requested.ro'), ('Requested READ-ONLY', 'KÉRT READ-ONLY'))
        self.assertIn("input_tokens", source)
        self.assertIn("cache_read_input_tokens", source)
        self.assertIn("total_cost_usd", source)
        self.assertIn("bridge_run_id", source)
        self.assertIn("running:t('state.running.short')", source)
        self.assertEqual(self.i18n('state.running.short'), ('RUNNING', 'FUT'))
        self.assertIn("'max-turn':'MAX-TURN'", source)
        self.assertIn("external_bridge_run_ids", HTML)
        self.assertIn("bridge_run_id", HTML)


class DashboardLanguageTests(unittest.TestCase):
    """The dictionary existing is not the same as the renderers using it.

    Every string the renderers emit used to be a Hungarian literal, so the
    language selector only ever translated the static shell.
    """

    def render_probe(self, language):
        helper = ModelRouterDashboardTests("test_lifecycle_description_uses_own_prompt_or_short_safe_fallback")
        source = helper.execution_source()
        probe = helper.i18n_runtime(language) + source + (
            "\nconsole.log(JSON.stringify({"
            "fallback:lifecycleDescription({}),"
            "external:t('exec.external'),"
            "running:t('state.running.short'),"
            "missing:t('no.such.key')"
            "}));"
        )
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        return json.loads(result.stdout)

    def test_renderer_output_follows_the_selected_language(self):
        english, hungarian = self.render_probe("en"), self.render_probe("hu")
        self.assertEqual(english["fallback"], "Internal router step")
        self.assertEqual(hungarian["fallback"], "Belső router-lépés")
        self.assertEqual(english["external"], "EXTERNAL")
        self.assertEqual(hungarian["external"], "KÜLSŐ")
        self.assertEqual(english["running"], "RUNNING")
        self.assertEqual(hungarian["running"], "FUT")

    def test_unknown_key_degrades_to_the_key_itself(self):
        # A missing key must never render as blank — it has to stay findable.
        self.assertEqual(self.render_probe("hu")["missing"], "no.such.key")

    def test_switching_language_rerenders_the_dynamic_content(self):
        # applyLanguage only walks [data-i18n] nodes; the tables and agent
        # cards are built by the renderers and need an explicit repaint.
        handler = HTML[HTML.index("document.getElementById('language-select').addEventListener"):]
        handler = handler[:handler.index("});")]
        self.assertIn("applyLanguage()", handler)
        self.assertIn("render()", handler)
        self.assertIn("renderAgents(agentActivity)", handler)


if __name__ == "__main__":
    unittest.main()


class RouterModuleImportTests(unittest.TestCase):
    """A normal Hermes install has no ``model_router`` package anywhere on
    sys.path: the repo is ``hermes-model-router`` and the plugin directory is
    ``model-router`` (hyphenated). _router_module() must still find the router
    by loading it from its own script directory."""

    def test_router_module_imports_without_a_parent_named_model_router(self):
        import subprocess
        import sys
        import tempfile

        script_dir = str(Path(web_viewer.__file__).resolve().parent)
        probe = (
            "import importlib.util, sys\n"
            f"sys.path.insert(0, {script_dir!r})\n"
            f"spec = importlib.util.spec_from_file_location('web_viewer', {str(Path(web_viewer.__file__).resolve())!r})\n"
            "web_viewer = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(web_viewer)\n"
            "router = web_viewer._router_module()\n"
            "assert router is not None, 'expected the router to import'\n"
            "assert hasattr(router, 'usage_guard'), 'expected the router module to carry usage_guard'\n"
            "print('OK')\n"
        )
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run(
                [sys.executable, "-c", probe],
                cwd=cwd,
                env={"HOME": cwd, "PATH": os.environ.get("PATH", "")},
                check=True, text=True, capture_output=True,
            )
        self.assertEqual(result.stdout.strip(), "OK")


class RouterStatusTests(unittest.TestCase):
    def test_router_status_degrades_to_empty_instead_of_failing(self):
        """The dashboard is a standalone script; it must keep serving the log
        even when the router package truly cannot be imported. _router_module()
        now has its own importlib fallback (see RouterModuleImportTests), so this
        drives the degrade path through that seam directly rather than via the
        old sys.modules trick, which the fallback would simply route around."""
        with patch.object(web_viewer, "_router_module", return_value=None):
            status = web_viewer._router_status()
        self.assertEqual(
            status, {"cooldowns": {}, "load": {}, "window_minutes": 0, "routable": []}
        )

    def test_only_a_routable_tier_is_offered_as_orchestrator(self):
        """A delegation-only target has no model entry in the router, so picking
        it raises a KeyError on the first routing decision. The card and the
        callability switch still apply to it -- only the orchestrator role does
        not."""
        status = web_viewer._router_status()
        self.assertNotIn("opus5", status["routable"])
        self.assertNotIn("sonnet5", status["routable"])
        self.assertIn("terra", status["routable"])
        renderer = HTML[HTML.index("const select=$('default-model-select');"):]
        self.assertIn("currentConfig.routable", renderer.split("}")[0] + renderer[:400])

    def test_router_status_reports_cooling_tiers_and_account_load(self):
        """Read through the router's own helpers rather than recomputed here, so
        the panel and the routing decision cannot disagree about availability."""
        status = web_viewer._router_status()
        self.assertIn("cooldowns", status)
        self.assertIn("load", status)
        self.assertIsInstance(status["cooldowns"], dict)
        self.assertIsInstance(status["load"], dict)
        for entry in status["cooldowns"].values():
            self.assertIn("seconds", entry)
            self.assertIn("reason", entry)

    def test_every_tier_the_router_knows_can_report_a_cooldown(self):
        """sonnet5 was missing from a hardcoded tuple here, so a cooling Sonnet
        reported nothing and the dashboard showed that account as merely idle —
        the one reading the operator most needs when Claude is the spare."""
        from unittest.mock import patch

        config = {
            "models": {"luna": "m", "terra": "m", "sol": "m", "qwen": "m"},
            "callable": {"luna": True, "terra": True, "sol": True,
                         "opus5": True, "sonnet5": True, "qwen": True},
            "usage_report": {"window_seconds": 3600},
        }
        with patch("model_router._load_config", return_value=config), \
             patch("model_router._read_cooldown_state",
                   return_value={"tiers": {"sonnet5": {"reason": "quota exhausted"}}}), \
             patch("model_router._tier_cooldown_remaining",
                   side_effect=lambda tier, cfg: 600.0 if tier == "sonnet5" else 0.0), \
             patch("model_router._recent_account_load", return_value={}):
            status = web_viewer._router_status()

        self.assertIn("sonnet5", status["cooldowns"])
        self.assertEqual(status["cooldowns"]["sonnet5"]["reason"], "quota exhausted")


class PreferenceSettingsTests(DashboardProbeMixin, unittest.TestCase):
    """The per-work-kind chain editor. The reordering logic runs in node, not in
    a Python re-implementation, so a bug in the shipped source fails here."""

    def _mutate(self, prefs, kind, index, act):
        """Run the real mutatePreference against a stub DOM and return the new prefs."""
        source = self.javascript_function("mutatePreference")
        probe = (
            "let currentConfig=" + json.dumps({"preferences": prefs}) + ";"
            "function renderSettings(){};function saveSettings(){};"
            + source
            + f"\nmutatePreference({json.dumps(kind)},{index},{json.dumps(act)});"
            "console.log(JSON.stringify(currentConfig.preferences));"
        )
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        return json.loads(result.stdout)

    def test_moving_an_entry_up_reorders_the_chain(self):
        self.assertEqual(
            self._mutate({"design": ["sol", "opus5", "terra"]}, "design", 1, "up"),
            {"design": ["opus5", "sol", "terra"]},
        )

    def test_moving_the_first_entry_up_is_a_no_op(self):
        self.assertEqual(
            self._mutate({"design": ["sol", "opus5"]}, "design", 0, "up"),
            {"design": ["sol", "opus5"]},
        )

    def test_moving_the_last_entry_down_is_a_no_op(self):
        self.assertEqual(
            self._mutate({"design": ["sol", "opus5"]}, "design", 1, "down"),
            {"design": ["sol", "opus5"]},
        )

    def test_removing_the_last_entry_drops_the_kind_entirely(self):
        """An empty list is not "no preference" to the router — the absent key is."""
        self.assertEqual(self._mutate({"design": ["sol"]}, "design", 0, "del"), {})

    def test_removing_one_of_several_keeps_the_rest_in_order(self):
        self.assertEqual(
            self._mutate({"code": ["sol", "terra", "luna"]}, "code", 1, "del"),
            {"code": ["sol", "luna"]},
        )

    def _render(self, config, language="en"):
        source = self.javascript_function("renderPreferences")
        labels = {m: m.upper() for m in
                  ("luna", "spark", "terra", "sol", "opus5", "sonnet5", "qwen")}
        probe = (
            self.i18n_runtime(language)
            + "let html='';const box={set innerHTML(v){html=v;},appendChild(el){html+=el.outerHTML||el.textContent;}};"
            "function $(id){return id==='pref-kinds'?box:null;}"
            "let currentConfig=" + json.dumps(config) + ";"
            "document={createElement:()=>({className:'',set innerHTML(v){this._h=v;},"
            "get outerHTML(){return this._h||'';},textContent:''})};"
            + source
            + f"\nrenderPreferences({json.dumps(labels)});console.log(html);"
        )
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        return result.stdout

    def test_a_delegation_only_target_is_marked_apart_from_a_routed_one(self):
        """A purple chip means "handed to the conductor", not "routed here"."""
        html = self._render({
            "work_kinds": ["design"], "preferences": {"design": ["opus5", "sol"]},
            "routable": ["luna", "spark", "terra", "sol"],
            "callable": {"opus5": True, "sol": True},
        })
        self.assertIn("pref-chip external", html)
        self.assertIn("1.", html)

    def test_a_kind_without_a_preference_says_the_built_in_route_applies(self):
        html = self._render({
            "work_kinds": ["review"], "preferences": {},
            "routable": ["terra"], "callable": {"terra": True},
        })
        self.assertIn("pref-empty", html)

    def test_a_switched_off_model_is_not_offered(self):
        """Offering it would let you configure a chain entry that can never run."""
        html = self._render({
            "work_kinds": ["code"], "preferences": {},
            "routable": ["terra", "sol"], "callable": {"terra": True, "sol": False, "qwen": False},
        })
        self.assertIn('value="terra"', html)
        self.assertNotIn('value="sol"', html)

    def test_every_work_kind_has_a_label_in_both_languages(self):
        from model_router import WORK_KINDS

        for kind in WORK_KINDS:
            english, hungarian = self.i18n(f"kind.{kind}")
            self.assertTrue(english.strip(), kind)
            self.assertTrue(hungarian.strip(), kind)
            self.assertNotEqual(english, hungarian, f"kind.{kind} is untranslated")


class HermesFallbackChainTests(DashboardProbeMixin, unittest.TestCase):
    """The chains live in Hermes's config, not the router's — so the editor has to be
    careful with a file it does not own, and the UI has to say which file it writes."""

    OPTIONS = [
        {"key": "opus5", "provider": "anthropic", "model": "claude-opus-5"},
        {"key": "sonnet5", "provider": "anthropic", "model": "claude-sonnet-5"},
        {"key": "qwen", "provider": "qwen-token", "model": "qwen3.7-plus"},
    ]

    def test_a_route_outside_the_configured_targets_is_rejected(self):
        """Offering a route the installation lacks would configure a leaf that cannot run."""
        chain, error = web_viewer._clean_fallback_chain(
            [{"provider": "evil", "model": "x"}], self.OPTIONS)
        self.assertIsNone(chain)
        self.assertIn("Unknown route", error)

    def test_an_incomplete_entry_is_rejected(self):
        chain, error = web_viewer._clean_fallback_chain([{"provider": "anthropic"}], self.OPTIONS)
        self.assertIsNone(chain)
        self.assertIn("needs a provider and a model", error)

    def test_duplicates_collapse_and_order_is_kept(self):
        chain, error = web_viewer._clean_fallback_chain([
            {"provider": "qwen-token", "model": "qwen3.7-plus"},
            {"provider": "anthropic", "model": "claude-opus-5"},
            {"provider": "qwen-token", "model": "qwen3.7-plus"},
        ], self.OPTIONS)
        self.assertIsNone(error)
        self.assertEqual([e["model"] for e in chain], ["qwen3.7-plus", "claude-opus-5"])

    def test_an_empty_chain_is_preserved_not_dropped(self):
        """For a delegated worker [] means "no fallback" — not "inherit the parent's"."""
        chain, error = web_viewer._clean_fallback_chain([], self.OPTIONS)
        self.assertIsNone(error)
        self.assertEqual(chain, [])

    def test_an_absent_chain_is_left_alone(self):
        self.assertEqual(web_viewer._clean_fallback_chain(None, self.OPTIONS), (None, None))

    def test_a_write_keeps_a_restore_point(self):
        """This file carries providers, approvals and the command allowlist."""
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "config.yaml"
            target.write_text("model:\n  default: gpt-5.6-terra\n", encoding="utf-8")
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target):
                web_viewer._write_hermes_config({"model": {"default": "changed"}})
            backups = list(Path(directory).glob("config.yaml.bak-router-*"))
            self.assertEqual(len(backups), 1)
            self.assertIn("gpt-5.6-terra", backups[0].read_text(encoding="utf-8"))
            self.assertIn("changed", target.read_text(encoding="utf-8"))

    def test_saving_one_chain_does_not_clear_the_other(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "config.yaml"
            target.write_text(
                "delegation:\n  fallback_providers:\n  - provider: anthropic\n    model: claude-opus-5\n",
                encoding="utf-8")
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target), \
                 patch.object(web_viewer, "_fallback_chain_options", return_value=self.OPTIONS):
                error = web_viewer._save_hermes_fallback(
                    {"orchestrator": [{"provider": "qwen-token", "model": "qwen3.7-plus"}]}, {})
            self.assertIsNone(error)
            written = web_viewer.yaml.safe_load(target.read_text(encoding="utf-8"))
            self.assertEqual(written["delegation"]["fallback_providers"][0]["model"], "claude-opus-5")
            self.assertEqual(written["fallback_providers"][0]["model"], "qwen3.7-plus")

    def test_an_unreadable_config_is_never_overwritten(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent.yaml"
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", missing), \
                 patch.object(web_viewer, "_fallback_chain_options", return_value=self.OPTIONS):
                error = web_viewer._save_hermes_fallback(
                    {"orchestrator": [{"provider": "qwen-token", "model": "qwen3.7-plus"}]}, {})
            self.assertIn("could not be read", error)
            self.assertFalse(missing.exists())

    def test_the_section_names_the_file_it_writes_in_both_languages(self):
        english, hungarian = self.i18n("settings.fb.file")
        for text in (english, hungarian):
            self.assertIn("~/.hermes/config.yaml", text)


class CooldownPillLayoutTests(DashboardProbeMixin, unittest.TestCase):
    """A long cooldown reason must stay inside its card.

    Observed 2026-09-09: "cooling down · 355m · model unavailable on this account"
    spilled out of the Spark card and pushed its switch onto the neighbouring one.
    The reason text grew when durable-unavailability cooldowns were added, and the
    pill was pinned to a single line.
    """

    def _rule(self, selector):
        import re

        match = re.search(re.escape(selector) + r"\{([^}]*)\}", HTML)
        self.assertIsNotNone(match, f"{selector} has no rule")
        return match.group(1)

    def test_the_pill_may_wrap(self):
        rule = self._rule(".cooldown-pill")
        self.assertNotIn("white-space:nowrap", rule)
        self.assertIn("overflow-wrap:anywhere", rule)

    def test_the_pill_cannot_exceed_the_card(self):
        self.assertIn("max-width:100%", self._rule(".cooldown-pill"))

    def test_the_label_column_is_allowed_to_shrink(self):
        """Without min-width:0 a flex item never shrinks below its content, which is
        what pushed the switch out rather than wrapping the text."""
        self.assertIn("min-width:0", self._rule(".toggle-label"))

    def test_the_switch_keeps_its_size(self):
        self.assertIn("flex:0 0 44px", self._rule(".switch"))


class SettingsLabelTests(DashboardProbeMixin, unittest.TestCase):
    def test_the_default_model_says_what_only_it_controls(self):
        """It reads as redundant next to the preference chains unless it names the
        one thing a chain cannot change: the model Hermes itself starts on."""
        english, hungarian = self.i18n("settings.default.desc")
        self.assertIn("starts on", english)
        self.assertIn("indul", hungarian)

    def test_the_preference_chains_say_what_they_do_not_change(self):
        english, hungarian = self.i18n("settings.prefs.sub")
        self.assertIn("does not change the model Hermes starts on", english)
        self.assertIn("indulási modelljét nem", hungarian)


class DefaultModelSaveTests(unittest.TestCase):
    """Saving the Settings tab must not move the model Hermes starts on.

    The page posts ``default_model`` on every save -- a callable toggle, a
    preference chain, a fallback edit -- so an unrelated save used to rewrite
    Hermes's own ``model.default`` back onto this provider's tier. One click
    undid a Claude-parent setup, and the write left no restore point.
    """

    CONFIG = {
        "models": {"luna": "gpt-5.6-luna", "terra": "gpt-5.6-terra", "sol": "gpt-5.6-sol"},
        "callable": {"luna": True, "terra": True, "sol": False, "opus5": True},
        "fallbacks": {"sol": "terra"},
        "tier_providers": {
            "luna": "openai-codex", "terra": "openai-codex",
            "sol": "openai-codex", "opus5": "anthropic",
        },
        "default_model": "terra",
    }
    HERMES = "model:\n  default: claude-opus-5\n  provider: anthropic\n"

    def _config(self):
        import copy

        return copy.deepcopy(self.CONFIG)

    def _hermes_file(self, directory):
        target = Path(directory) / "config.yaml"
        target.write_text(self.HERMES, encoding="utf-8")
        return target

    def test_saving_without_changing_the_default_model_leaves_hermes_alone(self):
        config = self._config()
        with tempfile.TemporaryDirectory() as directory:
            target = self._hermes_file(directory)
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target):
                error = web_viewer._save_default_model("terra", config)
            self.assertIsNone(error)
            self.assertEqual(target.read_text(encoding="utf-8"), self.HERMES)
            self.assertEqual(list(Path(directory).glob("config.yaml.bak-router-*")), [])

    def test_changing_the_default_model_writes_it_through_with_a_restore_point(self):
        config = self._config()
        with tempfile.TemporaryDirectory() as directory:
            target = self._hermes_file(directory)
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target):
                error = web_viewer._save_default_model("luna", config)
            self.assertIsNone(error)
            self.assertEqual(config["default_model"], "luna")
            written = web_viewer.yaml.safe_load(target.read_text(encoding="utf-8"))
            self.assertEqual(written["model"]["default"], "gpt-5.6-luna")
            self.assertEqual(written["model"]["provider"], "openai-codex")
            self.assertEqual(written["model"]["api_mode"], "codex_responses")
            backups = list(Path(directory).glob("config.yaml.bak-router-*"))
            self.assertEqual(len(backups), 1)
            self.assertIn("claude-opus-5", backups[0].read_text(encoding="utf-8"))

    def test_a_disabled_tier_still_resolves_through_the_fallback_chain(self):
        config = self._config()
        config["default_model"] = "luna"
        with tempfile.TemporaryDirectory() as directory:
            target = self._hermes_file(directory)
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target):
                error = web_viewer._save_default_model("sol", config)
            self.assertIsNone(error)
            self.assertEqual(config["default_model"], "terra")
            written = web_viewer.yaml.safe_load(target.read_text(encoding="utf-8"))
            self.assertEqual(written["model"]["default"], "gpt-5.6-terra")

    def test_a_chain_with_no_enabled_tier_is_refused(self):
        config = self._config()
        config["callable"]["terra"] = False
        with tempfile.TemporaryDirectory() as directory:
            target = self._hermes_file(directory)
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target):
                error = web_viewer._save_default_model("sol", config)
            self.assertIn("No enabled fallback", error)
            self.assertEqual(config["default_model"], "terra")
            self.assertEqual(target.read_text(encoding="utf-8"), self.HERMES)

    def test_a_delegation_target_is_refused_rather_than_blanking_the_model(self):
        """opus5 has no entry in `models`, so the old code wrote model.default: ''
        and the router's own _decision would raise KeyError on the tier."""
        config = self._config()
        with tempfile.TemporaryDirectory() as directory:
            target = self._hermes_file(directory)
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target):
                error = web_viewer._save_default_model("opus5", config)
            self.assertIsNotNone(error)
            self.assertIn("opus5", error)
            self.assertEqual(config["default_model"], "terra")
            self.assertEqual(target.read_text(encoding="utf-8"), self.HERMES)


class ConfigPathTests(unittest.TestCase):
    """The dashboard must read the config file the router actually loads.

    The path was hardcoded to ~/.hermes/plugins/model_router/, but `hermes
    plugins install` creates model-router (the manifest name), so on a normal
    install the dashboard read nothing and every save raised.
    """

    def test_the_dashboard_reads_the_file_beside_it(self):
        self.assertEqual(
            web_viewer.CONFIG_PATH,
            Path(web_viewer.__file__).resolve().parent / "router_config.yaml",
        )

    def test_the_dashboard_and_the_router_agree_on_one_file(self):
        import model_router

        self.assertEqual(web_viewer.CONFIG_PATH, model_router._CONFIG_PATH)


class HaikuDashboardTests(DashboardProbeMixin, unittest.TestCase):
    """Claude delegation adds a third Claude tier; the dashboard must show it
    wherever it shows the other two, or Haiku workers are counted nowhere."""

    def test_haiku_is_styled_like_sonnet5(self):
        import re

        css = "".join(re.findall(r"<style>(.*?)</style>", HTML, re.S))
        selectors = {
            tier: sorted(
                match.group(1).replace(tier, "<tier>")
                for match in re.finditer(r"([^{};]*\.%s[^{};]*)\{" % tier, css)
            )
            for tier in ("sonnet5", "haiku")
        }
        self.assertTrue(selectors["haiku"], "expected haiku to carry tier styling")
        self.assertEqual(selectors["haiku"], selectors["sonnet5"])

    def test_haiku_is_enumerated_everywhere_sonnet5_is(self):
        for fragment in ("--haiku:", 'class="card haiku"',
                         ".pill.haiku{color:var(--haiku)}", ".task-tree-marker.haiku{background:var(--haiku)"):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, HTML)
        self.assertEqual(HTML.count("'sonnet5'"), HTML.count("'haiku'"))
        self.assertEqual(HTML.count("sonnet5:"), HTML.count("haiku:"))

    def test_haiku_is_offered_by_the_tier_filter(self):
        # Since Task 7, the static #tier select only has the "all" option; the
        # per-tier <option>s (haiku included) come from tierFilterOptions(),
        # grouped by account.
        source = self.javascript_function("tierFilterOptions")
        probe = source + "\nconsole.log(tierFilterOptions([{account:'anthropic',label:'Claude',tiers:['haiku','sonnet5','opus5']}]));"
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        self.assertIn("<option>haiku</option>", result.stdout)

    def test_haiku_has_labels_in_both_languages(self):
        for key in ("card.haiku", "model.desc.haiku"):
            with self.subTest(key=key):
                english, hungarian = self.i18n(key)
                self.assertTrue(english and hungarian)

    def test_a_haiku_call_is_classified_as_haiku(self):
        source = self.execution_source()
        parent = {"children": [{"id": "external", "model": "claude-haiku-4-5-20251001",
                                "routed_calls": [{"tier": "haiku", "model": "claude-haiku-4-5-20251001",
                                                  "effort": "external"}], "children": []}]}
        probe = (source + "\nfunction sessionIdFromTurn(entry){return String(entry?.turn_id||'').split(':')[0]}\n"
                 "const scope=executionScope([{tier:'terra'}],[]," + json.dumps(parent) + ",'haiku');"
                 "console.log(JSON.stringify({kinds:scope.nodes.map(executionKind),"
                 "summary:executionSummary([scope.calls])}));")
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        observed = json.loads(result.stdout)
        self.assertEqual(observed["kinds"], ["haiku"])
        self.assertEqual(observed["summary"]["haiku"], 1)


class AccountsApiTests(unittest.TestCase):
    """The dashboard's per-account view: usage, guard limits, and delegation state,
    the same shape for every account the router can delegate to."""

    def _build_config(self, directory):
        state_path = Path(directory) / "usage-state.json"
        log_path = Path(directory) / "claude-delegation.jsonl"
        fetched_at = time.time() - 60
        state_path.write_text(json.dumps({
            "anthropic": {
                "weekly": 13, "session": 5,
                "weekly_resets_at": "2026-09-24T16:00:00+00:00",
                "session_resets_at": None,
                "fetched_at": fetched_at,
            },
        }), encoding="utf-8")
        log_path.write_text("\n".join([
            json.dumps({"event": "registration", "registered": True, "reason": ""}),
            json.dumps({"event": "delegate_claude", "session_id": "s1", "turn_id": "t1", "outcome": "ran"}),
            json.dumps({"event": "delegate_claude", "session_id": "s1", "turn_id": "t2", "outcome": "refused"}),
        ]), encoding="utf-8")
        config = {
            "callable": {
                "luna": True, "terra": True, "sol": True,
                "haiku": True, "sonnet5": True, "opus5": True,
                # qwen deliberately has no entry at all here (not even False):
                # a tier with a `callable` key, on or off, still gets a card and
                # a switch (see test_a_switched_off_tier_still_appears_with_its_switch);
                # only a tier the config never mentions is truly absent.
            },
            "tier_providers": {
                "luna": "openai-codex", "terra": "openai-codex", "sol": "openai-codex",
                "haiku": "anthropic", "sonnet5": "anthropic", "opus5": "anthropic",
                "qwen": "qwen-token",
            },
            "usage_guard": {
                "state_path": str(state_path),
                "accounts": {
                    "anthropic": {"soft_percent": 70, "hard_percent": 90, "step_down": {"opus5": "sonnet5"}},
                    "openai-codex": {"soft_percent": 70, "hard_percent": 90, "step_down": {"sol": "terra"}},
                },
            },
            "claude_delegation": {
                "enabled": True,
                "default_tier": "sonnet",
                "log_path": str(log_path),
            },
        }
        return config, state_path, log_path

    def setUp(self):
        import model_router

        self.model_router = model_router
        model_router.usage_guard._reset_cache()

    def tearDown(self):
        self.model_router.usage_guard._reset_cache()

    def test_accounts_status_covers_every_account_with_a_callable_tier(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self._build_config(directory)
            with patch.object(web_viewer, "_router_module", return_value=self.model_router):
                accounts = web_viewer._accounts_status(config)

        # Qwen has no callable tier, so it is absent entirely.
        self.assertEqual(set(accounts.keys()), {"openai-codex", "anthropic"})

        claude = accounts["anthropic"]
        self.assertEqual(claude["label"], "Claude")
        self.assertEqual(claude["state"], "open")
        self.assertEqual(claude["usage"]["weekly"], 13)
        self.assertTrue(50 <= claude["usage_age_seconds"] <= 120)
        self.assertIs(claude["guard"], True)
        self.assertEqual(claude["soft_percent"], 70)
        self.assertEqual(claude["delegation"], {
            "tool": "delegate_claude",
            "enabled": True,
            "registered": True,
            "restart_needed": False,
            "default_tier": "sonnet",
            "tiers": list(self.model_router.claude_delegation.TIERS),
        })

        codex = accounts["openai-codex"]
        self.assertIsNone(codex["usage"])
        self.assertEqual(codex["state"], "unknown")
        self.assertEqual(codex["delegation"], {"tool": "delegate_task", "always_on": True})

    def test_a_switched_off_tier_still_appears_with_its_switch(self):
        """The shipped config ships spark:false. Dropping it from `tiers` when it
        is off meant the switch that would turn it back on vanished with it."""
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self._build_config(directory)
            config["callable"]["spark"] = False
            config["tier_providers"]["spark"] = "openai-codex"
            with patch.object(web_viewer, "_router_module", return_value=self.model_router):
                accounts = web_viewer._accounts_status(config)
        self.assertIn("spark", accounts["openai-codex"]["tiers"])

    def test_an_account_with_every_tier_switched_off_still_gets_a_card(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self._build_config(directory)
            for tier in ("haiku", "sonnet5", "opus5"):
                config["callable"][tier] = False
            with patch.object(web_viewer, "_router_module", return_value=self.model_router):
                accounts = web_viewer._accounts_status(config)
        self.assertIn("anthropic", accounts)
        self.assertEqual(set(accounts["anthropic"]["tiers"]), {"haiku", "sonnet5", "opus5"})

    def test_accounts_status_never_fetches_usage(self):
        from unittest.mock import MagicMock

        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self._build_config(directory)
            anthropic_fetcher = MagicMock()
            codex_fetcher = MagicMock()
            with patch.dict(self.model_router.usage_guard.FETCHERS,
                             {"anthropic": anthropic_fetcher, "openai-codex": codex_fetcher}), \
                 patch.object(web_viewer, "_router_module", return_value=self.model_router):
                web_viewer._accounts_status(config)
            anthropic_fetcher.assert_not_called()
            codex_fetcher.assert_not_called()

    def test_save_usage_limits_updates_and_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self._build_config(directory)

            error = web_viewer._save_usage_limits(
                {"anthropic": {"soft_percent": 60, "hard_percent": 85}}, config
            )
            self.assertIsNone(error)
            self.assertEqual(config["usage_guard"]["accounts"]["anthropic"]["soft_percent"], 60)
            self.assertEqual(config["usage_guard"]["accounts"]["anthropic"]["hard_percent"], 85)

            self.assertIsNotNone(web_viewer._save_usage_limits(
                {"anthropic": {"soft_percent": 90, "hard_percent": 85}}, config))
            self.assertIsNotNone(web_viewer._save_usage_limits(
                {"anthropic": {"soft_percent": 10, "hard_percent": 150}}, config))
            self.assertIsNotNone(web_viewer._save_usage_limits(
                {"anthropic": {"soft_percent": 0, "hard_percent": 90}}, config))
            self.assertIsNotNone(web_viewer._save_usage_limits(
                {"nope": {"soft_percent": 10, "hard_percent": 90}}, config))

    def test_save_claude_delegation_updates_and_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self._build_config(directory)

            error = web_viewer._save_claude_delegation(
                {"enabled": False, "default_tier": "haiku"}, config
            )
            self.assertIsNone(error)
            self.assertEqual(config["claude_delegation"]["enabled"], False)
            self.assertEqual(config["claude_delegation"]["default_tier"], "haiku")

            self.assertIsNotNone(web_viewer._save_claude_delegation({"default_tier": "gpt"}, config))
            self.assertIsNotNone(web_viewer._save_claude_delegation({"enabled": "yes"}, config))

    def test_read_delegation_log_separates_audits_from_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self._build_config(directory)
            audits, registration = web_viewer._read_delegation_log(config)
            self.assertEqual(len(audits), 2)
            self.assertEqual({a["outcome"] for a in audits}, {"ran", "refused"})
            self.assertTrue(all(a["session_id"] == "s1" for a in audits))
            self.assertEqual(registration, {"event": "registration", "registered": True, "reason": ""})

    def test_read_delegation_log_only_reads_the_tail(self):
        """The log is never rotated, so a full read every /api/entries poll would slow down
        forever. Only the last _DELEGATION_LOG_TAIL_BYTES bytes are ever loaded."""
        with tempfile.TemporaryDirectory() as directory:
            config, _, log_path = self._build_config(directory)
            registration = json.dumps({"event": "registration", "registered": True, "reason": ""})
            early_audits = [
                json.dumps({"event": "delegate_claude", "session_id": f"early-{i}",
                            "turn_id": f"t{i}", "outcome": "ran"})
                for i in range(50)
            ]
            late_audits = [
                json.dumps({"event": "delegate_claude", "session_id": f"late-{i}",
                            "turn_id": f"t{i}", "outcome": "ran"})
                for i in range(5)
            ]
            log_path.write_text("\n".join([registration] + early_audits + late_audits), encoding="utf-8")

            with patch.object(web_viewer, "_DELEGATION_LOG_TAIL_BYTES", 400):
                audits, tail_registration = web_viewer._read_delegation_log(config)

            # Only lines from the tail come back: the early lines (and the registration,
            # which precedes them) fall outside the 400-byte window.
            self.assertTrue(audits, "expected at least one audit line from the tail")
            session_ids = {a["session_id"] for a in audits}
            self.assertTrue(session_ids.issubset({f"late-{i}" for i in range(5)}))
            self.assertFalse(session_ids & {f"early-{i}" for i in range(50)})
            self.assertIsNone(tail_registration)
            # Every returned line parsed cleanly -- a seek into the middle of a line would
            # have produced a JSON error that is silently skipped, not a corrupt entry.
            for audit in audits:
                self.assertEqual(audit["event"], "delegate_claude")

    def test_read_delegation_log_skips_a_non_utf8_line(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, log_path = self._build_config(directory)
            good_line = json.dumps({"event": "delegate_claude", "session_id": "s2",
                                     "turn_id": "t9", "outcome": "ran"}).encode("utf-8")
            bad_line = b"\xff\xfe not valid utf-8 \x80\x81"
            log_path.write_bytes(good_line + b"\n" + bad_line + b"\n")

            audits, registration = web_viewer._read_delegation_log(config)

            self.assertEqual(len(audits), 1)
            self.assertEqual(audits[0]["session_id"], "s2")
            self.assertIsNone(registration)

    def test_usage_refresh_endpoint_reads_through_the_shared_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self._build_config(directory)
            config_path = Path(directory) / "router_config.yaml"
            with open(config_path, "w", encoding="utf-8") as f:
                web_viewer.yaml.dump(config, f)

            reading = self.model_router.usage_guard.Reading(
                weekly=20.0, session=4.0,
                weekly_resets_at="2026-09-25T00:00:00+00:00", session_resets_at=None,
                fetched_at=time.time(),
            )

            def fake_read(account, cfg):
                # Mimics the real read()'s cache side effect, without a network call.
                self.model_router.usage_guard._slot(account)["reading"] = reading
                return reading

            with patch.object(web_viewer, "CONFIG_PATH", config_path), \
                 patch.object(self.model_router.usage_guard, "read", side_effect=fake_read) as mocked_read:
                server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{server.server_port}/api/usage/refresh?account=anthropic",
                        method="POST",
                    )
                    with urllib.request.urlopen(request) as response:
                        status = response.status
                        payload = json.load(response)
                    self.assertEqual(status, 200)
                    self.assertEqual(payload["account"]["usage"]["weekly"], 20.0)
                    mocked_read.assert_called_once()

                    bad_request = urllib.request.Request(
                        f"http://127.0.0.1:{server.server_port}/api/usage/refresh?account=nope",
                        method="POST",
                    )
                    try:
                        urllib.request.urlopen(bad_request)
                        self.fail("expected HTTPError for unknown account")
                    except urllib.error.HTTPError as exc:
                        self.assertEqual(exc.code, 400)
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)


class AccountCardTests(DashboardProbeMixin, unittest.TestCase):
    """Settings shows one account card per account, not a flat toggle list plus a
    separate load box. Every card has the same five rows: models, usage, limits,
    delegation, load."""

    CODEX_INFO = {
        "label": "Codex",
        "tiers": ["luna", "terra", "sol"],
        "state": "unknown",
        "usage": None,
        "usage_age_seconds": None,
        "has_usage_source": False,
        "guard": True,
        "soft_percent": 70,
        "hard_percent": 90,
        "step_down": {"sol": "terra"},
        "delegation": {"tool": "delegate_task", "always_on": True},
    }

    CLAUDE_INFO = {
        "label": "Claude",
        "tiers": ["haiku", "opus5", "sonnet5"],
        "state": "open",
        "usage": {
            "weekly": 13, "session": 5,
            "weekly_resets_at": "2026-09-24T16:00:00+00:00",
            "session_resets_at": None,
        },
        "usage_age_seconds": 60,
        "has_usage_source": True,
        "guard": True,
        "soft_percent": 70,
        "hard_percent": 90,
        "step_down": {"opus5": "sonnet5"},
        "delegation": {
            "tool": "delegate_claude", "enabled": True, "registered": True,
            "restart_needed": False, "default_tier": "sonnet",
            "tiers": ["haiku", "sonnet", "opus"],
        },
    }

    NO_USAGE_SOURCE_INFO = {
        "label": "Qwen",
        "tiers": ["qwen"],
        "state": "unknown",
        "usage": None,
        "usage_age_seconds": None,
        "has_usage_source": False,
        "guard": False,
        "soft_percent": None,
        "hard_percent": None,
        "step_down": {},
        "delegation": {"tool": "delegate_task", "always_on": True},
    }

    def _account_functions(self):
        return "\n".join(self.javascript_function(name)
                          for name in ("ageText", "resetText", "usageRow", "accountCard"))

    def _run(self, script):
        # 'status.locale' must resolve to a real BCP-47 tag: toLocaleString throws
        # on the stub's usual echo-the-key behaviour.
        probe = "const t=k=>k==='status.locale'?'en-US':k;\n" + self._account_functions() + "\n" + script
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        return result.stdout.strip()

    def _card(self, account, info, extra_config=None):
        config = {"callable": {}, "cooldowns": {}, "load": {"openai-codex": 3, "anthropic": 1},
                   "window_minutes": 60, "accounts": {account: info}}
        if extra_config:
            config.update(extra_config)
        script = (f"let currentConfig={json.dumps(config)};"
                  f"console.log(accountCard({json.dumps(account)},currentConfig.accounts[{json.dumps(account)}]));")
        return self._run(script)

    def test_markup_has_the_new_containers_not_the_old_ones(self):
        start = HTML.index('id="settings-panel"')
        end = HTML.index("</section>", start)
        panel = HTML[start:end]
        self.assertIn('id="account-cards"', panel)
        self.assertNotIn('id="callable-toggles"', HTML)
        self.assertNotIn('id="account-load"', HTML)

    def test_every_card_has_the_same_row_structure(self):
        import re

        codex_card = self._card("openai-codex", self.CODEX_INFO)
        claude_card = self._card("anthropic", self.CLAUDE_INFO)
        expected = ["models", "usage", "limits", "delegation", "load"]
        self.assertEqual(re.findall(r'class="account-row (\w+)"', codex_card), expected)
        self.assertEqual(re.findall(r'class="account-row (\w+)"', claude_card), expected)

    def test_claude_card_specifics(self):
        registered = self._card("anthropic", self.CLAUDE_INFO)
        self.assertIn('data-account-toggle="anthropic"', registered)
        self.assertIn('<select data-default-tier', registered)
        for tier in ("haiku", "sonnet", "opus"):
            self.assertIn(f'value="{tier}"', registered)
        self.assertIn("account.delegation.live", registered)

        restart_info = dict(self.CLAUDE_INFO,
                             delegation=dict(self.CLAUDE_INFO["delegation"], restart_needed=True))
        restarting = self._card("anthropic", restart_info)
        self.assertIn("account.delegation.restart", restarting)

    def test_codex_card_specifics(self):
        codex_card = self._card("openai-codex", self.CODEX_INFO)
        self.assertIn("account.delegation.always", codex_card)
        self.assertNotIn("data-account-toggle", codex_card)

    def test_an_account_without_a_usage_source_shows_the_none_message_and_disabled_limits(self):
        card = self._card("qwen-token", self.NO_USAGE_SOURCE_INFO)
        self.assertIn("account.usage.none", card)
        self.assertIn('data-limit="soft" data-account="qwen-token" value="" disabled', card)
        self.assertIn('data-limit="hard" data-account="qwen-token" value="" disabled', card)

    def test_usage_row_places_soft_and_hard_ticks(self):
        script = "console.log(usageRow('account.usage.week',55,null,70,90));"
        row = self._run(script)
        self.assertIn('style="left:70%"', row)
        self.assertIn('style="left:90%"', row)

    def test_every_i18n_key_used_by_the_new_code_exists_in_both_languages(self):
        for key in [
            "settings.accounts.heading", "account.state.open", "account.state.soft",
            "account.state.closed", "account.state.unknown", "account.models",
            "account.usage", "account.usage.week", "account.usage.session",
            "account.usage.resets", "account.usage.age", "account.usage.none",
            "account.usage.refresh", "account.limits", "account.limits.soft",
            "account.limits.hard", "account.limits.stepdown", "account.delegation",
            "account.delegation.via", "account.delegation.always",
            "account.delegation.default_tier", "account.delegation.live",
            "account.delegation.restart", "account.load", "account.load.calls",
            "settings.routing.heading",
        ]:
            self.i18n(key)

    def test_save_payload_includes_usage_limits_and_claude_delegation(self):
        source = self.javascript_function("saveSettings")
        self.assertIn("usage_limits:", source)
        self.assertIn("claude_delegation:", source)


class AccountGroupTests(DashboardProbeMixin, unittest.TestCase):
    """The main view groups the per-model count cards by account, with a compact
    weekly usage bar per group, and the #tier filter mirrors the same grouping."""

    TIER_ACCOUNTS = {
        "luna": "openai-codex", "spark": "openai-codex", "terra": "openai-codex",
        "sol": "openai-codex", "haiku": "anthropic", "sonnet5": "anthropic",
        "opus5": "anthropic", "qwen": "qwen-token",
    }
    ACCOUNTS = {
        "openai-codex": {"label": "Codex"},
        "anthropic": {"label": "Claude"},
        "qwen-token": {"label": "Qwen"},
    }

    def _group_source(self):
        start = HTML.index("const ACCOUNT_ORDER=")
        end = HTML.index("function accountCard(account,info)")
        return HTML[start:end]

    def _run(self, script, language="en"):
        probe = self.i18n_runtime(language) + "\n" + self.javascript_function("resetText") + "\n" \
            + self.javascript_function("ageText") + "\n" + self.javascript_function("usageRow") + "\n" \
            + self._group_source() + "\n" + script
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        return result.stdout.strip()

    def test_accounts_are_ordered_codex_then_claude_then_others(self):
        out = self._run(
            "console.log(JSON.stringify(accountGroupsFor("
            + json.dumps(self.TIER_ACCOUNTS) + "," + json.dumps(self.ACCOUNTS) + ")));"
        )
        groups = json.loads(out)
        self.assertEqual(
            [(g["account"], g["tiers"]) for g in groups],
            [
                ("openai-codex", ["luna", "spark", "terra", "sol"]),
                ("anthropic", ["haiku", "sonnet5", "opus5"]),
                ("qwen-token", ["qwen"]),
            ],
        )

    def test_an_account_with_no_configured_tiers_is_omitted(self):
        accounts = dict(self.ACCOUNTS, **{"unused-account": {"label": "Unused"}})
        out = self._run(
            "console.log(JSON.stringify(accountGroupsFor("
            + json.dumps(self.TIER_ACCOUNTS) + "," + json.dumps(accounts) + ")));"
        )
        groups = json.loads(out)
        self.assertNotIn("unused-account", [g["account"] for g in groups])

    def test_claude_group_is_filtered_to_tiers_present_in_tier_accounts(self):
        tier_accounts = dict(self.TIER_ACCOUNTS)
        del tier_accounts["opus5"]
        out = self._run(
            "console.log(JSON.stringify(accountGroupsFor("
            + json.dumps(tier_accounts) + "," + json.dumps(self.ACCOUNTS) + ")));"
        )
        groups = {g["account"]: g["tiers"] for g in json.loads(out)}
        self.assertEqual(groups["anthropic"], ["haiku", "sonnet5"])

    def test_compact_usage_shows_the_weekly_bar_with_ticks_and_state(self):
        info = {
            "state": "soft", "has_usage_source": True, "soft_percent": 70, "hard_percent": 90,
            "usage": {"weekly": 75, "weekly_resets_at": None, "session": 12},
            "usage_age_seconds": 90,
        }
        out = self._run(f"console.log(compactUsage({json.dumps(info)}));")
        self.assertIn('account-usage soft', out)
        self.assertIn('style="left:70%"', out)
        self.assertIn('style="left:90%"', out)
        self.assertIn('75%', out)

    def test_compact_usage_with_no_usage_source_shows_the_none_message(self):
        out = self._run(f"console.log(compactUsage({json.dumps({'has_usage_source': False})}));")
        english, _ = self.i18n("main.usage.none")
        self.assertIn(english, out)

    def test_tier_filter_options_are_grouped_by_account_label(self):
        out = self._run(
            "console.log(tierFilterOptions(accountGroupsFor("
            + json.dumps(self.TIER_ACCOUNTS) + "," + json.dumps(self.ACCOUNTS) + ")));"
        )
        self.assertIn('<optgroup label="Codex">', out)
        self.assertIn('<optgroup label="Claude">', out)
        self.assertIn('<optgroup label="Qwen">', out)
        codex_start = out.index('<optgroup label="Codex">')
        claude_start = out.index('<optgroup label="Claude">')
        qwen_start = out.index('<optgroup label="Qwen">')
        self.assertLess(claude_start, qwen_start, "Qwen must come after Claude")
        codex_block = out[codex_start:claude_start]
        claude_block = out[claude_start:qwen_start]
        qwen_block = out[qwen_start:]
        for tier in ("luna", "spark", "terra", "sol"):
            with self.subTest(tier=tier):
                self.assertIn(f'<option>{tier}</option>', codex_block)
        for tier in ("sonnet5", "haiku", "opus5"):
            with self.subTest(tier=tier):
                self.assertIn(f'<option>{tier}</option>', claude_block)
        self.assertIn('<option>qwen</option>', qwen_block)

    def test_static_markup_still_has_every_tier_count_id(self):
        cards_start = HTML.index('class="cards"')
        cards_end = HTML.index('id="account-groups"', cards_start)
        cards_html = HTML[cards_start:cards_end]
        for tier in ("luna", "spark", "terra", "sol", "opus5", "sonnet5", "haiku", "qwen"):
            with self.subTest(tier=tier):
                self.assertIn(f'id="{tier}"', cards_html)

    def test_static_tier_select_has_only_the_all_option(self):
        start = HTML.index('<select id="tier">')
        end = HTML.index('</select>', start)
        select_html = HTML[start:end]
        self.assertNotIn('<option>', select_html)
        self.assertIn('data-i18n="router.tier.all"', select_html)

    def test_main_usage_none_exists_in_both_languages(self):
        english, hungarian = self.i18n("main.usage.none")
        self.assertEqual(english, "no usage data")
        self.assertEqual(hungarian, "nincs használati adat")


class DelegationChipTests(DashboardProbeMixin, unittest.TestCase):
    """Every prompt row gets one chip per delegated worker: Codex children come
    from routed calls already in the router log, Claude children come from the
    claude-delegation audit lines that /api/entries returns as `delegations`."""

    ACCOUNTS_STATE = {
        "accounts": {},
        "tier_accounts": {
            "terra": "openai-codex", "sol": "openai-codex",
            "haiku": "anthropic", "sonnet5": "anthropic", "opus5": "anthropic",
        },
    }

    DOM_SHIM = (
        "global.document={createElement(tag){return {tag,className:'',textContent:'',"
        "title:'',children:[],append(...els){this.children.push(...els)}};}};"
    )

    def _source(self):
        names = [
            "executionOwnCalls", "executionRawCall", "executionCalls", "executionKind",
            "sessionIdFromTurn", "tierAccount", "accountLabel", "assignDelegations",
            "chip", "stepDownHoverText", "delegationChips",
        ]
        return "\n".join(self.javascript_function(name) for name in names)

    def _run(self, accounts_state, script):
        probe = (
            self.DOM_SHIM
            + f"let accountsState={json.dumps(accounts_state)};"
            + self._source()
            + "\n" + script
        )
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        return result.stdout.strip()

    def test_assignment_by_session_and_time(self):
        """Two runs in the same session get the audit that landed after them but
        before the next one; an audit in an unrelated session goes nowhere."""
        runs = [
            {"first": {"turn_id": "s1:a", "timestamp": "2026-01-01T10:00:00Z"}},
            {"first": {"turn_id": "s1:b", "timestamp": "2026-01-01T10:05:00Z"}},
        ]
        audits = [
            {"session_id": "s1", "timestamp": "2026-01-01T10:02:00Z", "tag": "first"},
            {"session_id": "s1", "timestamp": "2026-01-01T10:06:00Z", "tag": "second"},
            {"session_id": "s2", "timestamp": "2026-01-01T10:03:00Z", "tag": "stray"},
        ]
        out = self._run(self.ACCOUNTS_STATE, (
            "const map=assignDelegations(" + json.dumps(runs) + "," + json.dumps(audits) + ");"
            "console.log(JSON.stringify([...map.entries()].map(([k,v])=>[k,v.map(a=>a.tag)])));"
        ))
        self.assertEqual(json.loads(out), [[0, ["first"]], [1, ["second"]]])

    def test_chips_for_both_accounts(self):
        run = {"scope": {"nodes": [
            {"kind": None, "routed_calls": [{"tier": "terra", "model": "terra"}], "children": []},
            {"kind": None, "routed_calls": [{"tier": "haiku", "model": "haiku"}], "children": []},
        ]}}
        audits = [
            {"tier_used": "haiku", "outcome": "ran"},
            {"tier_requested": "opus", "tier_used": "sonnet", "outcome": "lowered",
             "adjusted": "opus→sonnet (weekly usage 74%)"},
            {"tier_requested": "sonnet", "tier_used": "sonnet", "outcome": "refused",
             "message": "Claude delegation closed: …"},
        ]
        out = self._run(self.ACCOUNTS_STATE, (
            "const box=delegationChips(" + json.dumps(run) + "," + json.dumps(audits) + ");"
            "console.log(JSON.stringify(box.children.map(c=>({text:c.textContent,title:c.title}))));"
        ))
        chips = json.loads(out)
        self.assertEqual([c["text"] for c in chips],
                          ["Codex: terra", "Claude: haiku", "Claude: sonnet ↓", "Claude: sonnet ✕"])
        # M5: one hover format for both accounts -- "<from>→<to>, weekly N%".
        self.assertEqual(chips[2]["title"], "opus→sonnet, weekly 74%")
        # Refused/error chips keep their original message untouched.
        self.assertIn("Claude delegation closed", chips[3]["title"])

    def test_codex_step_down_chip_detects_the_appended_reason_clause(self):
        """Controller ruling: the router now appends the usage-limit clause onto
        the original reason rather than starting the reason with it, so the
        marker must be detected with a regex search, not startsWith."""
        run = {"scope": {"nodes": [
            {"kind": None, "routed_calls": [
                {"tier": "terra", "reason": "long work; usage soft limit: sol→terra (weekly 72%)"}
            ], "children": []},
        ]}}
        out = self._run(self.ACCOUNTS_STATE, (
            "const box=delegationChips(" + json.dumps(run) + ",[]);"
            "console.log(JSON.stringify(box.children.map(c=>({text:c.textContent,title:c.title}))));"
        ))
        self.assertEqual(json.loads(out), [{
            "text": "Codex: terra ↓",
            "title": "sol→terra, weekly 72%",
        }])

    def test_codex_hard_limit_reason_also_marks_a_step_down_chip(self):
        run = {"scope": {"nodes": [
            {"kind": None, "routed_calls": [
                {"tier": "terra", "reason": "usage hard limit: sol→terra (weekly 91%)"}
            ], "children": []},
        ]}}
        out = self._run(self.ACCOUNTS_STATE, (
            "const box=delegationChips(" + json.dumps(run) + ",[]);"
            "console.log(JSON.stringify(box.children.map(c=>({text:c.textContent,title:c.title}))));"
        ))
        self.assertEqual(json.loads(out), [{
            "text": "Codex: terra ↓",
            "title": "sol→terra, weekly 91%",
        }])

    def test_codex_hard_limit_session_window_reason_reformats_too(self):
        """M10 makes the router name the session window when that is what
        actually triggered the hard limit; the same hover formatter must
        handle that clause too."""
        run = {"scope": {"nodes": [
            {"kind": None, "routed_calls": [
                {"tier": "terra", "reason": "usage hard limit: sol→terra (session 95%)"}
            ], "children": []},
        ]}}
        out = self._run(self.ACCOUNTS_STATE, (
            "const box=delegationChips(" + json.dumps(run) + ",[]);"
            "console.log(JSON.stringify(box.children.map(c=>c.title)));"
        ))
        self.assertEqual(json.loads(out), ["sol→terra, session 95%"])

    def test_a_skipped_step_down_never_marks_the_chip(self):
        """I3: the router also writes this clause when the target itself is
        unavailable ('...skipped (terra unavailable)'); that must never be
        read as a real step-down."""
        run = {"scope": {"nodes": [
            {"kind": None, "routed_calls": [
                {"tier": "sol", "reason": "usage soft limit: sol→terra skipped (terra unavailable)"}
            ], "children": []},
        ]}}
        out = self._run(self.ACCOUNTS_STATE, (
            "const box=delegationChips(" + json.dumps(run) + ",[]);"
            "console.log(JSON.stringify(box.children.map(c=>c.textContent)));"
        ))
        self.assertEqual(json.loads(out), ["Codex: sol"])

    def test_claude_fallback_chip_with_no_audit_shows_the_tier_name_not_the_target(self):
        """No audits means the entry has to be read off the routed node itself,
        whose tier field is a router target name (sonnet5/opus5); the chip must
        still say the short tier name (sonnet/opus) the Settings tier selector
        and the audit log both use."""
        run = {"scope": {"nodes": [
            {"kind": None, "routed_calls": [{"tier": "sonnet5"}], "children": []},
            {"kind": None, "routed_calls": [{"tier": "opus5"}], "children": []},
        ]}}
        out = self._run(self.ACCOUNTS_STATE, (
            "const box=delegationChips(" + json.dumps(run) + ",[]);"
            "console.log(JSON.stringify(box.children.map(c=>c.textContent)));"
        ))
        self.assertEqual(json.loads(out), ["Claude: sonnet", "Claude: opus"])

    def test_haiku_chip_with_no_audit_log(self):
        run = {"scope": {"nodes": [
            {"kind": None, "routed_calls": [{"tier": "haiku"}], "children": []},
        ]}}
        out = self._run(self.ACCOUNTS_STATE, (
            "const box=delegationChips(" + json.dumps(run) + ",[]);"
            "console.log(JSON.stringify(box.children.map(c=>c.textContent)));"
        ))
        self.assertEqual(json.loads(out), ["Claude: haiku"])

    def test_delegation_chips_are_appended_to_the_routes_cell_not_the_header(self):
        """.router-run-header is a CSS grid with a fixed number of column tracks
        (redefined at the 1120px and 700px breakpoints, plus .no-details) — an
        extra header child auto-places onto a stray grid cell instead of
        flowing inline. The chips must attach to the existing router-run-routes
        cell instead, after its route pills."""
        renderer = HTML[HTML.rindex("render=function(){"):]
        self.assertIn("const delegationMap=assignDelegations(runData,delegations)", renderer)
        routes_start = renderer.index(
            "const routes=document.createElement('span');routes.className='router-run-routes';"
        )
        header_append = renderer.index(
            "header.append(dateEl,timeEl,prompt,stateEl,total,routes,workers);"
        )
        between = renderer[routes_start:header_append]
        self.assertIn("routes.append(delegationChips(", between)
        self.assertNotIn("header.append(delegationChips(", renderer)

    def test_delegation_chips_are_a_descendant_of_the_routes_cell_at_runtime(self):
        """Node probe over the live renderer's actual header-building statements
        (not a reimplementation): builds the same header/routes elements the
        real code builds and asserts the chips box lands inside routes.children,
        never directly in header.children."""
        renderer = HTML[HTML.rindex("render=function(){"):]
        start = renderer.index("const routes=document.createElement")
        end = renderer.index("if(hasDetails)header.addEventListener")
        segment = renderer[start:end]
        probe = (
            self.DOM_SHIM
            + "const header={children:[],append(...els){this.children.push(...els)}};"
            + "const dateEl='dateEl',timeEl='timeEl',prompt='prompt',stateEl='stateEl',total='total';"
            + "const t=k=>k;const accountingCalls=[];const workerCalls=0;"
            + "function appendRoutePills(el,calls){el.append('PILL')}"
            + "function delegationChips(record,audits){const b=document.createElement('span');b.className='delegation-chips';return b}"
            + "const record={id:'r'},delegationMap=new Map([[0,['audit']]]),runIndex=0;"
            + segment
            + "console.log(JSON.stringify({"
            + "routesHasChip:routes.children.some(c=>c&&c.className==='delegation-chips'),"
            + "headerHasChip:header.children.some(c=>c&&c.className==='delegation-chips'),"
            + "headerHasRoutes:header.children.includes(routes)}));"
        )
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        observed = json.loads(result.stdout)
        self.assertTrue(observed["routesHasChip"])
        self.assertFalse(observed["headerHasChip"])
        self.assertTrue(observed["headerHasRoutes"])
