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
  heavy: [terra, opus5, qwen, sonnet5]
  light: [luna, spark]
```

The contract names the groups, and an unavailable target is annotated with its
live replacement — `opus5 [unavailable for another 15 min; use qwen instead]` —
instead of vanishing from the list. Dropping it said only that it was gone;
naming the replacement is what turns one account's exhaustion into work
continuing somewhere else.

**"Comparable in strength" is quoted to the conductor verbatim, so a wrong
grouping is an instruction to misroute.** `sonnet5` sat in the light group until
1.9.5, next to a tier bounded at 700 characters and `low` effort. The conductor
duly substituted Luna for it whenever the Codex account looked loaded, and
"stabilize, correct, test and commit the dirty foundation" ran on Luna twelve
times. Group by what a target can actually carry, not by what it costs.

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

When both are offered the contract used to add "use `sonnet5` by default and
reserve `opus5` for consequential or hard work". That sentence dates from the
commit that made these targets reachable at all, when no preference mechanism
existed and a bare list of two names told the conductor nothing. One arrived 23
hours later and the older answer was never withdrawn, so two contradictory
instructions sat in the same paragraph — and the unconditional one beat the
hedged one every time. It is now emitted **only where the operator has configured
nothing**, and the check reads the configured chain rather than its currently
available winner, so a cooling `opus5` cannot revive the built-in default at the
one moment the operator's own order needs to be what speaks.

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
  delegation instruction instead — it reaches work through `delegate_task`, and
  the conductor is the one that has to honour it.

A kind with no chain keeps its built-in route, so configuring nothing changes
nothing. A kind with a chain overrides that route **completely**, including the
safety defaults that send design, security and deployment work to Sol. That is
deliberate: the operator owns the mapping.

**The whole order reaches the conductor, not just its winner.** Availability
folds in the cooldown, so naming only the first *available* entry meant a cooling
`opus5` erased `code` from the contract entirely — indistinguishable from a kind
nobody configured, and the conductor could not advance to an entry it was never
told existed. Cooling entries are annotated instead, and the conductor's own tier
belongs in the chain too: `code: opus5 > terra` rendered as `code: opus5` and
lost the very entry that has to take over.

```
The operator's target order per kind of work, highest priority first --
  design: sol > opus5; code: opus5 > terra; review: sonnet5 > terra.
A leaf of one of these kinds must take the first target in that kind's order,
and when an entry is marked unavailable must move to the next entry in the same
order rather than choosing freely.
```

**It is phrased as an instruction because advice loses.** It shares a paragraph
with the `[spark]` and `[sol]` rules, which are imperatives. Until 1.9.5 the
chain arrived as "Honour these when a leaf matches the kind and the target is
free" while an unconditional built-in default sat two sentences earlier — so
`code -> model:opus5` never once decided a leaf.

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
  sol_long: medium            # reached by length, not by choice
  explicit_sol: medium        # you asked for Sol
  explicit_sol_xhigh: high    # you asked for Sol and said how hard
  explicit_luna_xhigh: high   # ...and the same for every other tier
  explicit_terra_xhigh: high
```

`[<tier>:xhigh]` is the label form. It parsed for every tier from the start but
was honoured only for Sol, so `[luna:xhigh]` ran silently at Luna's floor with no
way to say otherwise; `explicit_<tier>` was unreadable config everywhere else for
the same reason. Both keys are general since 1.9.5.

Resolution walks from the most specific key to the plain tier, and **a key you
have not written changes nothing** — that is what makes adding a tier here
optional rather than a behaviour change. An escalation degrades to its tier's
`explicit_` key rather than to the tier's floor, so a missing
`explicit_<tier>_xhigh` means "no escalation configured" instead of silently
capping the request.

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

Note what that last condition rules out. A usage quota belongs to the account, and
the failure is recorded **before** a replacement is looked for — which benches
every sibling on that account first. So on a genuine quota exhaustion this
substitution cannot fire at all: the configured replacement is already cooling.
What survives here is the case it is actually good for — a transient blip, or a
model this account cannot use — where a same-account sibling is the right answer.
Moving quota-stopped work needs a different account, and that is a dispatch
decision, not a model swap.

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

### A worker stopped by a quota comes back as a re-dispatch

A leaf that dies on an account limit has not failed at its task, but nothing in
the delegation envelope says so. It reports the goal, the status and the
provider's error, and the conductor is left to guess whether to retry, re-plan or
drop — while re-sending the same goal to the same target fails identically until
the cooldown lapses.

Everything needed to answer that is already here: the cooldown state says which
accounts are refusing calls and for how long, the preference chain says what comes
next, and the classifier that decides every route can tell what kind of work the
goal is. So a delegation outcome carrying a quota-stopped task comes back with the
target named:

```
[ROUTER — A WORKER STOPPED ON AN ACCOUNT LIMIT]
- Implement and commit the backend portion of the reliability ledger.
  code work -> re-dispatch with model:opus5
Re-dispatch each one with the model: parameter named above and tell the retry to
continue from what the stopped worker already committed in its worktree instead
of starting over. Do not re-plan or narrow the goal: only the account changed.
```

When the whole chain is cooling it says what to wait for and for how long, so
waiting stays a legible option instead of a guess.

### A delegation that failed before any worker existed

The notice above reads a delegation *outcome*. Sometimes there is none: the tool
returns an error inline, no child ever runs, and nothing will be delivered later
to explain it.

The error is also not a fact about the account it names, and this one is worth
stating precisely because it is easy to get backwards. `delegate_task` resolves
the configured **default** delegation provider once for the whole call — at
`delegate_tool.py:496`, before `_normalize_task_list` has even parsed the tasks
— and returns `tool_error` if that fails. The per-task target is resolved much
later, inside the dispatch loop.

**So an unavailable default provider blocks every delegation, including a task
that names a target on a healthy account.** Its `model:` value is never read.
With Codex exhausted, a `model: "opus5"` task fails on the Codex quota although
Opus 5 runs on Anthropic and was never contacted:

```
Cannot resolve delegation provider 'openai-codex':
Codex provider quota exhausted (429); retry after 3731s.
```

The parent's own conclusion — "the Codex quota is out, so the opus5 delegation
failed" — is therefore literally correct, however wrong it sounds.

The notice says that, and deliberately does **not** advise a retry with a
different `model:`, which would loop. It separates the targets that are
themselves healthy from the fact that none of them is reachable, and names the
three ways out: repoint `delegation.provider`/`delegation.model` at a working
route, do the work in the turn, or wait.

The real fix is in the host: resolve the default lazily, or tolerate its failure
when every task names its own target.

**It names the target and stops there.** Re-dispatching by itself would be the
hardcoded selection this design exists to avoid: what to do with a stopped leaf —
retry, narrow, wait, drop — is the conductor's call.

Both outcome shapes are answered: the consolidated batch envelope, and the early
single-child notice that arrives while siblings are still running, which exists
precisely so the conductor can act then rather than at batch end. The off-provider
path answers too — a `code` chain starting with `opus5` puts the conductor itself
on Claude, and that branch returns early, so the setup that needs this most would
otherwise have been the one to miss it.

Two false positives are excluded by construction. An ordinary failure is left
alone, because only a limit is safe to re-send unchanged. And the reason is read
from the envelope's own status/error lines rather than from the whole block — a
worker whose subject *is* quota handling otherwise reports itself as
quota-stopped, which the leaves of this very plugin do.

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

The label still has to be true. A `[spark]` **or `[luna]`** leaf must actually be
read-only: one that writes is rejected, and one touching production, security,
credentials or payments escalates to Sol. Both are judged from the verbs,
independently of the subject matter — "identify the layout branches" is source
discovery, not design work.

Luna faced no check at all until 1.9.5, which is how a conductor handed it
"Stabilize, correct, test, and commit the dirty foundation now" and the router
obeyed. The write-verb list had a matching hole: `commit` was not in it, and
neither were any Hungarian imperatives, so the guard would have passed that goal
even once it existed.

Closing that hole opened another one, and the two changes were in the same
series. `commit` is also a noun, and the goal contract requires a goal to name
the commit it builds on — so "…at commit 7abc123" made a read-only source map
read as mutating, and a `[spark]` leaf was escalated off Spark for saying exactly
what it had been told to say. A commit *reference* is stripped before the
write-verb test; an instruction to commit still counts.

**`[opus5]` and `[sonnet5]` are not labels.** They read like `[sol]` and do the
opposite of what the writer meant: the override vocabulary knows only this
provider's four tiers, so the prefix is inert, the goal is classified on its
remaining text, and the leaf works to completion on the account the dispatcher
was trying to spare. A delegated leaf whose goal opens with one — while running
on one of this provider's models — is stopped at its first call, its tool use
switched off, and its single answer is the correction, which reaches the parent
as the leaf's own summary:

```
MISDISPATCHED: this goal names opus5 in its text, which is not a route.
Re-dispatch it unchanged with delegate_task(model="opus5").
```

Both facts are required. A leaf already on that account has a redundant prefix
rather than a wrong one, and a **root** turn carrying `[opus5]` is you asking for
Opus, not a dispatch bug. `[opus-review]` and `[sonnet-review]` are unaffected:
those are real labels, routed to the read-only CLI bridge.

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

That rule was here from the start and lost anyway, seven goals running, because it
shares a paragraph with `[spark]` and `[sol]` — which *are* prefixes. `[opus5]` is
the obvious blend of the two mechanisms. Since 1.10.1 the contract names the
mistake rather than restating the rule, and a leaf that makes it is stopped
instead of quietly becoming a Sol leaf.

Claude is one of those targets, and a Claude leaf must carry every fact it needs
in its goal — it does not share the conversation:

```python
{"goal": "Diagnose and fix the fullscreen calendar card in /path/to/repo. …",
 "model": "sonnet5"}
```

### A goal carries what the worker cannot see

That rule is not specific to Claude. **Every** delegated worker starts at
`history=0` on every target, so a fact the conductor knows and does not write
down is a fact the worker spends iterations rediscovering — against a budget it
cannot raise: `delegate_task` accepts a `max_iterations` argument and the host
ignores it, because `delegation.max_iterations` is authoritative "so budgets stay
predictable".

Measured on one Opus leaf: sixteen iterations, twenty tool calls (13 `terminal`,
6 `read_file`, 1 `search_files`), context grown from 20k to 56k, and **not one
edit**. The whole budget went on reconstructing a repository the goal never
described, because the goal was a product requirement:

> Implement a tenant-scoped, safe customer-profile merge capability for Booking
> SaaS: an authorized admin can review two duplicate customer profiles and merge
> a phone-only and email-only record into one canonical profile…

No path, no branch, no base commit, no files. The leaf that finished did so in
nine iterations with one write, and the only difference was its goal: it carried
its own state ("the user already ran the `ALTER USER` command, it succeeded") and
asked for a single artefact.

So the contract now requires every goal to state the absolute worktree path, the
branch and the commit it builds on, what already exists there, which files or
modules are in scope, and how the result is verified — and to give one worker one
finishable artefact rather than a feature to implement. A goal phrased as a
product requirement has no boundary, and it is spent before the first edit.

Raising `delegation.max_iterations` is the blunt instrument here, not the first
move: it is global, so it also widens every Codex leaf on the shared quota.

**These requirements ride on the `delegate_task` schema, not on the message.**
They used to travel only inside the forced preflight, so a turn that skipped it
delegated with nobody having been told what a goal must carry — and a root prompt
shorter than `orchestration.min_chars` (60) skips it, creating no conductor at
all. A twelve-character `inplementald` produced exactly that: a whole-feature
goal with no worktree, branch or base commit in it.

The schema is the right carrier because a middleware edit does not persist into
the conversation — that is why the preflight needs a rescue pass at all. The
parent may delegate on any call of the turn, so an appended sentence would have
to be repeated on every one of them; a tool description is read once, exactly
where the goal is written. When a preflight *does* fire, the conductor's contract
already carries the same rules and the schema is left alone.

**And `context` is required, because a description is advice.** Measured after
the description shipped and was live: the parent read it and dispatched
"read-only release readiness review of *the customer-profile merge worktree*"
anyway — no path — and the reviewer spent 31 shell commands over 16 iterations
without reaching a verdict. That was the third time a description lost, after
the built-in `sonnet5` tie-breaker and the `[opus5]` prefix rule.

So the per-task `context` — which the schema already describes as the place for
"file paths, error messages, constraints", and which genuinely reaches the child
— is moved into `required`, and its description says what belongs in it. A
context-free call is now invalid rather than merely discouraged, and on a tool
carrying `strict: true` the provider is the one enforcing it. This is the same
move the preflight already makes for `role` and `context` on a conductor, for
the reason its own comment gives: *natural-language instructions alone are not a
reliable control plane*.

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

**1.10.8** — A cooling `sonnet5` shows its cooldown in the dashboard: the status panel walked a hardcoded tier list that omitted it, so that account read as merely idle

**1.10.7** — A `delegate_task` that cannot resolve its provider comes back explaining that the host resolves the default route before it reads the tasks, so an exhausted default blocks even a task naming a healthy account — and that retrying with another `model:` would loop

**1.10.6** — A goal that names the base commit it builds on, as the contract requires, no longer reads as an instruction to commit: a read-only `[spark]` leaf was being escalated off Spark for complying

**1.10.5** — Per-task `context` is required on the delegate_task schema, with a description that says what belongs in it: the goal description alone was live, read, and ignored

**1.10.4** — A parent that dispatches twice inside the matching window no longer shows one worker twice under the wrong model while the other runs unlisted: a child session is matched by its goal and claimed once

**1.10.3** — The goal requirements ride on the `delegate_task` schema too, so a parent that never got a preflight — a root prompt under `orchestration.min_chars` creates no conductor — still writes goals that carry a worktree, a branch and a boundary

**1.10.2** — The contract requires a goal to carry what the worker cannot see — worktree, branch, base commit, scope, verification — after an Opus leaf spent all sixteen iterations rediscovering a repository its goal never described, and made no edit

**1.10.1** — A goal naming `opus5` or `sonnet5` in its text is stopped at its first call and returned for re-dispatch, instead of running to completion on Sol; the contract names that mistake rather than restating the rule

**1.10.0** — A worker stopped by an account limit comes back as a re-dispatch with the next target named, from both the batch envelope and the early single-child notice

**1.9.5** — Implementation work stops landing on Luna and on a hardcoded default: `sonnet5` leaves the light substitution group, `[luna]` faces the read-only check `[spark]` already had, `explicit_<tier>` and `[<tier>:xhigh]` work for every tier, and the operator's chain is stated in full as an instruction rather than as advice

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
