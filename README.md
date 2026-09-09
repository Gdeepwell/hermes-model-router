# Model Router — Hermes Agent Plugin

Routing and delegation for Hermes Agent. It keeps the user-facing conversation on one durable parent model, lets that parent's plan choose which model runs each delegated worker, and records every decision in a privacy-safe audit log.

The point of choosing per worker is that the models sit on **different accounts**: Codex (Luna/Spark/Terra/Sol), a Qwen token plan, and a Claude subscription (Opus 5/Sonnet 5). Spreading independent work across them spends separate quotas in parallel instead of draining one.

## Install

```bash
hermes plugins install Gdeepwell/hermes-model-router
```

Then enable it:

```bash
hermes plugins enable model-router
```

## Features

### Automatic Routing

The tier name is what you write in `model:` and in `router_config.yaml`. The
account is the point of the table: spreading independent work across the three
is what keeps a single quota from carrying everything.

| Tier | Model | Account | Purpose |
|------|-------|---------|---------|
| `terra` | GPT-5.6 Terra | Codex | Durable default parent and conductor |
| `luna` | GPT-5.6 Luna | Codex | Simple tasks, short answers |
| `spark` | GPT-5.3 Codex-Spark | Codex | Read-only code analysis, bounded subtasks |
| `sol` | GPT-5.6 Sol | Codex | Complex, security-sensitive, design |
| `qwen` | Qwen 3.7 Plus | Qwen token plan | Delegation target only |
| `opus5` | Claude Opus 5 | Claude subscription | Delegation target, off by default (see below) |
| `sonnet5` | Claude Sonnet 5 | Claude subscription | Delegation target, off by default (see below) |

`qwen`, `opus5` and `sonnet5` are delegation targets rather than routable tiers:
the middleware cannot move a call across providers, so they are reached by a
plan choosing them with `model:`, not by the router switching to them mid-turn.
Any tier in `models` can hold the orchestrator role, including one on another
account — that choice is made at spawn time, where the provider is still open.

### Substitution groups

`fallbacks` rewrites the model on a call whose provider and credentials are
already fixed, so it can only ever move work between tiers on one account.
Moving it between *accounts* has to happen where the provider is still being
chosen — when the plan picks a target. So peers are expressed to the conductor
rather than applied behind it:

```yaml
peer_groups:
  heavy: [terra, opus5, qwen]
  light: [luna, sonnet5, spark]
```

The contract names the groups, and an unavailable target is annotated with its
live replacement — `opus5 [unavailable for another 15 min; use qwen instead]` —
instead of vanishing from the list. Dropping it said only that it was gone;
naming the replacement is what turns one account's exhaustion into work
continuing somewhere else.

Substitution is for capacity, not permission. A `[spark]` leaf must still be
read-only wherever it runs, and design work still belongs to Sol — so a cooling
Sol is shown as unavailable rather than hidden, because waiting for it is a
legitimate answer and rerouting the work is not.

### Stable Parent Policy

The user-facing conversation stays on one durable parent model. Classifier results are worker recommendations, not silent parent switches. Sol, Spark, Qwen and Opus never become the apparent responder merely because a prompt mentions UI, CSS, design, or long text.

### Privacy-First Audit Logging

- 240-character bounded prompt preview
- Redacted sensitive data (tokens, URLs with credentials, etc.)
- Parent/child correlation IDs
- Token and cost metrics when supplied by the runtime
- No raw prompt or response bodies stored in logs

### Bounded Delegation

- Max 2 concurrent child agents
- Max 2 spawn depth: the parent delegates a conductor, the conductor delegates leaves, and leaves cannot delegate further
- Max 16 child iterations
- Handoff capsule required for every worker task

### Claude targets

Claude is reached like any other delegation target — `model: "opus5"` or
`model: "sonnet5"`, running on the native Anthropic Messages API. The child is
an ordinary worker with the usual tools, so it can implement rather than merely
read, and it runs in the background alongside the plan's other leaves. It draws
on a different subscription from every other target, which is the point.

```yaml
delegation:
  targets:
    opus5:
      provider: anthropic
      model: claude-opus-5
    sonnet5:
      provider: anthropic
      model: claude-sonnet-5
```


**These targets are on.** They shipped switched off for months because every
call returned:

```
HTTP 400 invalid_request_error
Third-party apps now draw from your extra usage, not your plan limits.
Add more at claude.ai/settings/usage and keep going.
```

That reads like a policy prohibition. It was not one, and it was not an account
limit either — the earlier diagnosis here ruled out the credential, request
size, tools, delegation, and the account's extra-usage setting, then wrongly
concluded the Hermes version was identical to upstream and could be excluded.

**The version was the whole difference.** Proved by running the same
`auth.json` and `HERMES_HOME` against two trees: `HTTP 400` on the old one,
`OK` on current upstream. The old tree authenticated with Hermes's own OAuth
app — a `manual:hermes_pkce` token in the credential pool — which the API
correctly classifies as a third-party app. Current upstream instead borrows the
Claude Code login (`agent/anthropic_credentials.py` plus the credential pool's
`_seed_from_singletons`), which is the same grant Claude Code itself uses.

After migrating to the current tree on 2026-09-09, the same account answers on
both paths — a direct `hermes -z --provider anthropic --model claude-opus-5`,
and the delegated credential path that `_task_credentials` resolves per child.
If you are on an older Hermes and see the 400, update before you buy credit.


Meanwhile the CLI bridge below does draw on the plan: `claude -p` *is* Claude
Code, so a `[sonnet-review]` or `[opus-review]` leaf works regardless. It is
read-only and replaces a single call instead of running an agent.

The router does not *route* these children — `route_llm_request` returns `None`
for a model outside its own tier map, so nothing here rewrites them — but it does
record them. Invisible to the router had meant invisible to the operator: a
Claude worker produced no card, no count and no line in the per-account load, so
the one account whose usage most needed watching was the one nothing reported on.
They now appear as `opus5` and `sonnet5` alongside the other tiers, and their
`callable` switches govern whether the conductor is offered them at all.

A read-only CLI bridge also exists (`[opus-review]` / `[sonnet-review]`,
`coding_agent.delegated_review`). It replaces a single call rather than running
an agent, so it cannot write and holds a child slot for the duration; the native
target above supersedes it for ordinary work.

### A parent on a fallback account still orchestrates

Rewriting a model is provider-bound; orchestrating is not. When Hermes's own
fallback chain moves the orchestrator onto another account — Codex out of quota,
so the parent continues on Sonnet — that parent is still the orchestrator and
still gets the preflight, the delegation contract and the `model:` parameter
contract. Its model is never rewritten, because this middleware cannot change a
request's provider; only the instructions are added.

Until 1.9.1 it did not: `route_llm_request` returned early for any model outside
its own provider, so a parent on a fallback account silently lost its contract
and worked alone. A second gate compounded it by recognising only Sol and
`default_model` as orchestrators.

The forced conductor follows the `code` preference chain when one is set, then
`default_model` when that tier is callable, then its `fallbacks` chain. Planning and
coordination are code work, and the conductor is not cheap: six consecutive
conductors on the Codex account each ran to the 16-iteration cap and spent 36% of a
five-hour limit before a leaf did any real work. Putting `opus5` first in `code`
moves the planning to another subscription and leaves the primary quota for the
work itself. Pinning it to `default_model` regardless was also how the preflight
used to fail exactly when it was needed — on the account that had just run out.

One more thing had to change for this to actually fire. An Anthropic OAuth request
is normalised for Claude Code compatibility, which renames every tool to
`mcp__<name>` — so a Claude parent was told it had no `delegate_task` tool and
skipped its preflight. The lookup accepts both names now, and the four places that
each carried their own copy of it share one function, which is why the mismatch
survived as long as it did.

`preflight_skipped` events record the tool names that were on offer, so "no
delegate_task tool" can be told apart from "no tools at all" and from "a name this
router does not recognise" without adding instrumentation after the fact.

### Default model vs. preference chains

These look redundant and are not. `default_model` decides five things; a
preference chain overrides exactly one of them:

| | overridden by a chain? |
|---|---|
| the general-purpose route when no gate matches | **yes**, by the `default` kind |
| which tier counts as an orchestrator at all | no |
| the tier the forced conductor runs on | no |
| whether the preflight dispatches | no |
| **the model Hermes itself starts on** (`model.default`, `model.provider`) | no |

The last row is the one that matters: a chain re-routes a turn, it does not change
which model the process launches with. Setting `default_model` in the dashboard
writes Hermes's own config as well as the router's.

### Preferred models per kind of work

Configured under `preferences:` in `router_config.yaml`, or from the Settings
tab, as an ordered chain per work kind — `design`, `code`,
`explore`, `review`, `sensitive`, `critical`, `long`, `chat`, `default`. The
router walks the chain and takes the first entry that is switched on.

A chain entry means one of two different things, and the dashboard colours them
differently because the difference is not cosmetic:

- **A tier of the router's own provider** (`luna`, `spark`, `terra`, `sol`) is a
  real route. The router rewrites the model and the chain also replaces the
  built-in fallback order for that kind.
- **Anything on another account** (`opus5`, `sonnet5`, `qwen`) cannot be routed
  to at all: `route_llm_request` runs after the provider is chosen, so it can only
  swap models inside one provider. Such an entry is passed to the conductor as a
  delegation recommendation instead — it reaches work through `delegate_task`.

A kind with no chain keeps its built-in route, so configuring nothing changes
nothing. A kind with a chain overrides that route **completely**, including the
safety defaults that send design, security and deployment work to Sol. That is
deliberate: the operator owns the mapping.

### Hermes fallback chains

The Settings tab also edits the two chains that live in **Hermes's** config rather
than the router's, because that is where a cross-account rescue is decided:

| Chain | Key in `~/.hermes/config.yaml` | Applies to |
|---|---|---|
| Orchestrator | `fallback_providers` | the main agent, when its own provider cannot serve |
| Delegated workers | `delegation.fallback_providers` | every child spawned by `delegate_task` |

The second is not optional in practice. A child pinned to a target — anything
spawned with `model: "opus5"` and friends — **never inherits the orchestrator's
chain**, so without its own it runs with no fallback at all: a quota-exhausted
leaf simply dies mid-task. An empty list there means "no fallback", which is a
different instruction from the key being absent.

Do not confuse either with the router's own `fallbacks:`, which substitutes tiers
*within* one provider and cannot cross accounts.

The picker offers only routes that exist as `delegation.targets`, so a chain
cannot name something the installation cannot run. Because this file is not the
router's — it also carries providers, approvals and the command allowlist — every
save first copies it to `config.yaml.bak-router-<timestamp>`, and a config that
cannot be read is refused rather than overwritten.

### Reasoning effort per tier

A route carries an effort level, not just a model name. `effort:` sets it per tier,
and separately for the cases where the same tier means different work:

```yaml
effort:
  luna: low
  terra: medium
  sol: medium
  sol_long: medium          # reached by length, not by choice
  explicit_sol: medium      # you asked for Sol
  explicit_sol_xhigh: high  # you asked for Sol and said how hard
```

`[sol:xhigh]` is the label form — the only override that carries an effort with it.

### Images force a vision-capable route

Spark is text-only. An attached image routes to Terra as a policy decision, ahead
of every Spark path: label, benchmark override and tool loop alike. A `[spark]`
tag on a message with an image is not honoured, because the alternative is a leaf
that cannot see what it was asked about.

Historical image attachments are stripped from replayed context rather than resent.

### Quota fallback

Distinct from `fallbacks`, which handles a *disabled* tier. `quota_fallbacks`
handles a tier that answered `429` mid-turn:

```yaml
quota_fallbacks:
  spark:
    model: luna
    effort: medium
```

It fires once per turn and only if the replacement is itself callable, so an
exhausted Spark continues on Luna instead of failing the turn.

### A model this account cannot use

Three different unavailabilities, three different mechanisms — the distinction is
what decides whether a leaf survives:

| Condition | Recognised by | What happens |
|---|---|---|
| Switched off in the dashboard | `callable: false` | the `fallbacks` chain, at routing time |
| Quota exhausted (429) | account/weekly quota wording | `quota_fallbacks`, then a 15-minute cooldown |
| Provider blip (5xx) | 500/502/503/504, connection resets | a hardcoded substitute, then a short cooldown |
| **Refused outright (400/404)** | *"is not supported when using…"*, *"does not exist or you do not have access"* | the configured `fallbacks` chain, then a 6-hour cooldown |

The last row was a gap until 1.8.1. A tier switched **on** but refused by the
provider — `The 'gpt-5.3-codex-spark' model is not supported when using Codex with
a ChatGPT account.` — matched no runtime failover, so the leaf simply aborted.

It deliberately walks the **configured** `fallbacks` chain rather than the
hardcoded transient map: if you wrote `spark: luna`, a Spark that does not exist
on your account belongs on Luna, not wherever the blip handler would have put it.
The cooldown is long because nothing about this recovers by waiting; tune it with
`cooldown.unavailable_seconds`. Design and image guards still apply — an
unavailable model is no reason to break a routing policy.

An ordinary malformed-request 400 is untouched and still raises.

### Preempted tiers in the log

A route entry records not only the tier that won but the ones that independently
qualified and lost (`vetoed_by`). Without it a tier that never fires looks exactly
like a tier whose preconditions are never met — the log cannot tell you whether a
gate is dead or merely outranked.

### Shadow benchmarking

Off by default. When enabled, a turn can be run a second time on a forced tier and
both outcomes recorded, so a routing rule can be judged on results rather than on
the argument that produced it:

```yaml
shadow:
  enabled: false
  limit: 0
  path: ~/.hermes/logs/spark-shadow-benchmark.jsonl
```

`MODEL_ROUTER_BENCHMARK_FORCE_MODEL=<tier>` forces a single run from the
environment. The Spark read-only and image restrictions still apply — a benchmark
may not route work somewhere policy forbids.

### Retrying a failed downstream request

The plugin can re-issue one failed provider call through Hermes's `llm_execution`
middleware (`retry_call`), which is what lets a route that fails on arrival be
answered by a different model without failing the turn. It needs a Hermes new
enough to pass `retry_call` to execution middleware.

### Live Dashboard

```bash
python3 ~/.hermes/plugins/model_router/web_viewer.py
# http://localhost:8765
```

Three tabs: **Model Router**, **Settings**, and an embedded **Hermes Command
Center**. The interface is available in English and Hungarian.

The router tab opens with a counter card per tier — including the two Claude
tiers, whose children the router records without routing them — then routing
decisions grouped by prompt, each expandable into its individual API calls, with
a grouped/raw toggle, tier filters and search.

Below that, the live agent tree. It reads Hermes's durable delegation registry
and the `turn_lifecycle` table, and nests running and recent children under their
parent session with a privacy-safe task preview, state and age. Each row carries
`own N · total N` — the calls that agent made itself, and the calls its whole
subtree made — so a conductor that is quietly doing the work instead of
delegating it (`own 21 · total 21`) is visible at a glance, as is a leaf that
correctly does not delegate further (`own 9 · total 9`). The pills beside it name
the tiers those calls went to; an `external` pill means the router observed the
call rather than routing it, because the model belongs to another provider.

Everything refreshes every 3 seconds.

The Settings tab holds the callability switches, the default orchestrator, the
per-work-kind preference chains described above, and the interface language. It
shows availability alongside the switches. A tier that is
enabled but cooling carries a pill with the remaining time and the reason, since
the switch alone would not explain why traffic went elsewhere; underneath, the
per-account call counts for the same window the conductor is given. Both are read
through the router's own helpers rather than recomputed, so the panel and the
routing decision cannot disagree.

The server binds to `127.0.0.1` only, so it is not reachable from the local
network. Every setting on that tab writes to `router_config.yaml` and takes effect
on the **next routed call, with no restart** — the config is re-read on every
decision rather than cached. Changing the plugin's *code* does need a restart of
the Hermes process that loaded it.

One caveat: those writes go through a plain YAML dump, so **comments in
`router_config.yaml` are lost** the first time you save from the dashboard. Keep
anything you need to remember in this README rather than in the config.

## Configuration

The plugin loads `router_config.yaml` from the plugin directory automatically.

### Key Settings

```yaml
# Model availability
callable:
  luna: true
  spark: false   # disabled here; [spark] leaves follow the `fallbacks` chain
  terra: true
  sol: true
  opus5: true
  sonnet5: true
  qwen: true

# Which account each tier spends. Claude tiers are delegation targets rather
# than routable tiers, but they are still counted and switched here.
tier_providers:
  luna: openai-codex
  spark: openai-codex
  terra: openai-codex
  sol: openai-codex
  opus5: anthropic
  sonnet5: anthropic
  qwen: qwen-token

# Default parent model
default_model: terra

# Preferred models per kind of work, best first (see the section above).
# Unset kinds keep their built-in route.
preferences:
  review: [sonnet5, terra]
  design: [sol, opus5]

# Delegation limits (in ~/.hermes/config.yaml)
delegation:
  max_concurrent_children: 2
  max_spawn_depth: 2
  max_iterations: 16

# Audit logging
logging:
  prompt_preview_chars: 240
  redact_prompt_preview: true

# Orchestration: the parent hands the objective to a conductor, which
# plans the work and delegates the leaves.
orchestration:
  enabled: true
  min_chars: 60        # too short to decompose; skip the planner round trip
  max_tasks: 2         # must not exceed delegation.max_concurrent_children
  rescue_min_calls: 6  # a turn this deep with no worker gets one late checkpoint

# Benching a tier that just refused. quota_seconds is only the fallback for a
# provider that does not say when the allowance returns.
cooldown:
  enabled: true
  quota_seconds: 900          # no reset hint in the error
  quota_max_seconds: 21600    # ceiling for a provider-stated reset
  unavailable_seconds: 21600  # a 400/404 "model not supported" refusal
  allowed_fails: 3            # repeated non-quota failures within the window
  failure_window_seconds: 60
  failure_seconds: 60
```

`max_tasks` above `max_concurrent_children` is a hard error, not a partial
run: `delegate_task` rejects the whole batch. Keep them equal.

Declining to dispatch is logged too. `terra-spark-orchestration.jsonl` records
`preflight_forced` when a conductor is created and `preflight_skipped` — with
the gate that rejected it — when one is not, so a turn that ran twenty calls
with no worker says why.

### Usage reporting

Spreading work across accounts used to be an instruction with nothing behind it:
the conductor was told to use separate accounts, but could not see that one had
taken every call for the last hour and another had taken none. The routing
contract now carries a live figure read from the router's own log:

```
Recent load over the last 60 minutes, in calls per account: openai-codex 82.
These are call counts from this router's own log, not quota readings — read
them as relative load. qwen-token has taken none in this window.
```

Call counts, deliberately: the runtime does not report tokens or cost to the
route log, so a percentage would be invented. The sentence states the count and
stops there — "no calls" reads as spare capacity, but it is equally what an
exhausted account looks like, and an earlier version of this line recommended an
account whose weekly quota had already run out. Only the tail of the log is
parsed, since it reaches tens of megabytes and this runs on the preflight path.

A cooling target is annotated in the same sentence rather than dropped from the
list. LiteLLM excludes a deployment that is over its limit, but its deployments
are interchangeable and these are not — hiding a cooling Sol would invite the
planner to send design work somewhere the classifier then refuses outright.

```yaml
usage_report:
  enabled: true
  window_seconds: 3600
```

### Cooldowns

A tier that just rejected a call for quota is not a candidate for the next one.
A 429 puts it in cooldown; repeated failures inside a window do the same.

```yaml
cooldown:
  enabled: true
  path: ~/.hermes/state/model-router-cooldowns.json
  quota_seconds: 900
  allowed_fails: 3
  failure_window_seconds: 60
  failure_seconds: 60
```

Failures are recorded whatever provider served them. The execution middleware
returns early off-provider because it rewrites `request["model"]` within one
provider, but noticing that an account refused a call needs none of that — and
skipping it meant a weekly-quota 429 on the second account left no cooldown at
all, while the load report kept describing it as the one with no traffic.

The state is a file rather than process memory because the interactive TUI and
the gateway are separate processes — a note kept in memory would not be seen by
the one that needs it. A cooling tier is simply not callable, so the existing
fallback chain and the policy rule below both apply with no extra wiring: a
preference route moves on, a policy route says which tier is cooling and for
how long.

A quota cooldown lasts as long as the provider says, not as long as the config
guesses. Codex answers a usage-limit 429 with `resets_in_seconds` / `resets_at`;
that value wins, capped by `cooldown.quota_max_seconds` (6h) so a malformed hint
cannot bench a tier for a day. `cooldown.quota_seconds` remains the fallback for
providers that say nothing.

This matters more than it sounds. A three-hour account reset benched for the
configured 15 minutes produces a loop: the cooldown lapses, the tier is offered
again, and the next leaf spends its retries rediscovering the same wall.

A usage quota also belongs to the **account**, not the model. `tier_providers`
says which account each tier spends, so one tier's quota 429 benches its siblings
for the same duration — all four Codex tiers together, or Opus and Sonnet
together. Targets on other accounts are untouched, which is the point: the
planner should be reaching for them.

### Policy routes do not fall back

`fallbacks` exists for preference: a long request prefers Sol for capacity, and
demoting it to Terra is a quality trade. But some routes are policy — design work
reaches Sol because *only* Sol may do it, and consequential work escalates there
for the same reason. Satisfying those from the fallback chain would perform the
work on the tier the rule exists to keep it away from, precisely when Sol is out
of quota and the rule matters most.

Such a decision is marked at the point it is made and declines the chain, so a
disabled Sol fails loudly instead of quietly landing design work on Terra.

This is the *default*. A preference chain configured for that work kind replaces
it — see above — because the operator asked to own the mapping. A single-entry
chain (`sensitive: [sol]`) keeps the loud failure; add a second entry only if you
would rather the work continue elsewhere than stop.

## Usage

### Explicit Model Override

Prefix your message with a tag:

```
[luna] Simple question
[sol] Complex security analysis
[opus] Diagnostic review through the standalone bridge
```

A root turn is still assessed for you: design work reaches Sol whatever label
you type, and `[spark]` on a root turn defers to the orchestrator rather than
sending user-facing work to a read-only worker.

### Labels inside a plan

The same labels mean something stronger on a delegated worker, because there
the label was written by a conductor that saw the objective, the repository and
any screenshot — a better-informed decision than a keyword test on the goal
text. So a plan label is authoritative, and the design gate does not re-judge it.

The label still has to be true. A `[spark]` leaf must actually be read-only:
one that writes is rejected, and one touching production, security, credentials
or payments escalates to Sol. Both are judged from the verbs, independently of
the subject matter — "identify the layout branches" is source discovery, not
design work.

The check looks for contradiction, not corroboration. A leaf that names no write
verb passes, because the conductor already declared it read-only by labelling it;
requiring a second positive signal would let a hand-written verb list overrule
that declaration on phrasing alone. A root `[spark]` is a label someone typed
with nothing behind it, and there the stricter form still applies: it has to show
its read-only intent.

### Delegation

The parent delegates independent bounded subtasks, and picks the route for
each one with `model`:

```python
delegate_task(tasks=[
  {"goal": "[spark] Read-only source discovery for the calendar renderer.",
   "model": "luna"},
  {"goal": "[sol] Diagnose and fix the card layout.",
   "model": "sol"},
])
```

`model` is what actually selects the route; the enum is built from
`delegation.targets` in `~/.hermes/config.yaml`, filtered to tiers that are
currently callable. `callable` and `delegation.targets` are separate switches, so
without that filter a planner can pick a tier the cross-provider guard then
refuses mid-session, producing a leaf that never runs. A goal-text prefix only renames
the model *inside the default provider*, so it cannot reach a target on another
account — a leaf meant for Qwen must carry `model: "qwen"`.

Claude is one of those targets, and a Claude leaf must carry every fact it needs
in its goal — it does not share the conversation:

```python
{"goal": "Diagnose and fix the fullscreen calendar card in /path/to/repo. …",
 "model": "sonnet5"}
```

## Diagnosing a parent that will not delegate

`~/.hermes/logs/terra-spark-orchestration.jsonl` records why a preflight did not
run. Read its tail first — the answer is usually one field:

```bash
tail -5 ~/.hermes/logs/terra-spark-orchestration.jsonl \
  | jq '{event, parent_model, skip_reason, tools_seen}'
```

`preflight_forced` means the contract went out and the parent was asked to
delegate; what it does next is the model's decision. `preflight_skipped` names
the gate instead:

| `skip_reason` | Meaning |
|---|---|
| `orchestration_disabled` | `orchestration.enabled: false` |
| `tier_not_orchestrator:<tier>` | neither Sol, `default_model`, nor a delegation target |
| `subagent_turn` | already a delegated child; children do not orchestrate |
| `no_delegate_task_tool` | no delegation tool in the request — `tools_seen` lists what was there |
| `sol_preflight_disabled` | `sol_opus5_preflight.enabled: false` |
| `delegation_completion_delivery` | the turn is delivering a finished child's result |

`tools_seen` exists because `no_delegate_task_tool` reads identically whether the
request had no tools at all, the wrong wire shape, or a name the router did not
recognise — and those need different fixes. Establishing that distinction took a
live probe before the field was added.

If the parent *is* orchestrating and the work still lands on one account, read the
routing log instead: an `external delegation target` line means the router observed
the call rather than routing it, which is normal for Claude and Qwen.

## Policy

1. **Stable Parent** — The user-facing conversation does not silently switch models.
2. **Delegation is an exception, not the default** — Only for genuinely independent subtasks.
3. **Worker limits enforced** — Max 2 concurrent children, 2 spawn depth, 16 iterations.
4. **Privacy-safe audit** — 240-char bounded preview, redacted sensitive data.
5. **The plan decides the route** — On a delegated worker the conductor's label wins; the router enforces only what the label claims (read-only, non-consequential).
6. **Documented changes** — Every policy change updates README and tests.

## Tests

```bash
cd ~/.hermes/plugins/model_router
pytest
```

## Version

**1.9.4** — The forced conductor follows the `code` preference chain, so planning does not have to sit on the primary quota

**1.9.3** — A long cooldown reason wraps inside its card instead of displacing the switch; the settings labels say what only they control

**1.9.2** — A Claude parent's `mcp__`-prefixed delegate_task is recognised, so the preflight it was granted in 1.9.1 actually fires

**1.9.1** — A parent moved onto a fallback account keeps its delegation contract, the forced conductor follows the callable chain, and the contract no longer claims Claude is unavailable

**1.9.0** — Hermes's orchestrator and delegated-worker fallback chains are editable from Settings, with a restore point before every write

**1.8.2** — Quota cooldowns last as long as the provider says and cover every tier on that account, instead of 15 minutes on the one tier that happened to ask

**1.8.1** — A model the account cannot use (400/404 refusal) now takes the configured fallback chain and a long cooldown instead of aborting the leaf

**1.8.0** — Preferred models per kind of work as an ordered chain, configurable from Settings; the reference now also documents effort levels, vision routing, quota fallback, preempted tiers, shadow benchmarking and the agent tree

**1.7.0** — Claude tiers on: the 400 was a Hermes version authenticating with its own OAuth app, not an account or plan limit

**1.6.2** — Claude delegation targets ship off pending an account-side 400; what was ruled out, and how to check it, is documented rather than guessed at (superseded by 1.7.0)

**1.6.0** — Substitution groups across accounts, cooling targets annotated with their replacement, orchestrator selector restricted to routable tiers

**1.5.0** — Claude reached natively as a delegation target on subscription OAuth, Claude tiers counted and switchable like any other

**1.4.0** — Cooldowns after quota and repeated failures, per-account load in the routing contract, policy routes that decline the fallback chain, cooling tiers and load shown in the dashboard

**1.3.0** — Per-worker route selection across accounts, delegated Claude review leaves, plan labels authoritative on delegated workers, orchestration preflight with a late rescue and logged skip reasons

**1.2.0** — Stable parent policy, bounded delegation, privacy-safe logging

## License

MIT

## Author

SENTINEL — Hermes Agent Model Router
