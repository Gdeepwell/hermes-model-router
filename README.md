# Model Router — Hermes Agent Plugin

Routing and delegation for Hermes Agent. It keeps the user-facing conversation on one durable parent model, lets that parent's plan choose which model runs each delegated worker, and records every decision in a privacy-safe audit log.

The point of choosing per worker is that the models sit on **different accounts**: Codex (Luna/Spark/Terra/Sol), a Qwen token plan, and Claude through its own CLI. Spreading independent work across them spends separate quotas in parallel instead of draining one.

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

| Tier | Purpose |
|------|---------|
| **Qwen** | Separate account; reachable as a delegation target |
| **Luna** | Simple tasks, short answers, basic questions |
| **Spark** | Read-only code analysis, bounded coding subtasks |
| **Terra** | Durable default orchestrator, general-purpose tasks |
| **Sol** | Complex, security-sensitive, critical infrastructure |
| **Claude (Opus / Sonnet)** | Read-only review leaves through the Claude CLI, plus a standalone diagnostic bridge |

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

### Claude review leaves

Claude cannot be a `delegation.targets` entry: a delegation target is a
provider/model pair a child's tool loop runs *on*, and the Claude Code OAuth
credential is not usable from a third-party tool, so the only way to reach
Claude is to hand a task to its own CLI. It is an external agent, not a model.

The integration is therefore an execution swap. A delegated leaf whose goal
begins `[sonnet-review]` or `[opus-review]` has its single LLM call replaced by
a `claude -p` subprocess, and the verdict becomes that leaf's answer — so the
work draws on the Claude subscription instead of the Codex account, while still
running in parallel with the plan's other leaves.

```yaml
coding_agent:
  delegated_review:
    enabled: true
    models: [opus, sonnet]
```

Deliberately separate from `coding_agent.enabled`, which also arms a
label-free coding classifier that would capture the first call of a coding turn.
Delegated workers only — a root turn is never diverted into a subprocess.
Read-only (`--tools Read`, 16 turns, $5, 600s); writing is not offered because
parallel leaves share one working tree.

### Live Dashboard

```bash
python3 ~/.hermes/plugins/model_router/web_viewer.py
# http://localhost:8765
```

Routing decisions grouped by prompt, each expandable into its individual API
calls, with a grouped/raw toggle, tier filters and search. The Agents panel
reads Hermes's durable delegation registry and shows running and recent child
jobs nested under their parent session, with a privacy-safe task preview, state,
age, selected model and call count. Everything refreshes every 3 seconds.

The server binds to `127.0.0.1` only, so it is not reachable from the local
network. Model callability and the default orchestrator can be changed from the
Settings tab; those writes land in `router_config.yaml` and take effect on the
next routed call, with no restart needed.

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
  qwen: true

# Default parent model
default_model: terra

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
```

`max_tasks` above `max_concurrent_children` is a hard error, not a partial
run: `delegate_task` rejects the whole batch. Keep them equal.

Declining to dispatch is logged too. `terra-spark-orchestration.jsonl` records
`preflight_forced` when a conductor is created and `preflight_skipped` — with
the gate that rejected it — when one is not, so a turn that ran twenty calls
with no worker says why.

### Policy routes do not fall back

`fallbacks` exists for preference: a long request prefers Sol for capacity, and
demoting it to Terra is a quality trade. But some routes are policy — design work
reaches Sol because *only* Sol may do it, and consequential work escalates there
for the same reason. Satisfying those from the fallback chain would perform the
work on the tier the rule exists to keep it away from, precisely when Sol is out
of quota and the rule matters most.

Such a decision is marked at the point it is made and declines the chain, so a
disabled Sol fails loudly instead of quietly landing design work on Terra.

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
`delegation.targets` in `~/.hermes/config.yaml`. A goal-text prefix only renames
the model *inside the default provider*, so it cannot reach a target on another
account — a leaf meant for Qwen must carry `model: "qwen"`.

A read-only review leaf can go to Claude instead, which needs no `model`
because its route is the label:

```python
{"goal": "[sonnet-review] Review the pending calendar diff in /path/to/repo. Report only."}
```

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

**1.3.0** — Per-worker route selection across accounts, delegated Claude review leaves, plan labels authoritative on delegated workers, orchestration preflight with a late rescue and logged skip reasons

**1.2.0** — Stable parent policy, bounded delegation, privacy-safe logging

## License

MIT

## Author

SENTINEL — Hermes Agent Model Router
