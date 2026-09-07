# Model Router — Hermes Agent Plugin

Intelligent LLM routing for Hermes Agent. Routes between Qwen (orchestrator), Luna (simple tasks), Spark (read-only coding), Terra (default orchestrator), Sol (complex/security-critical), and Claude Opus 5 bridge — with stable parent policy, bounded delegation, and privacy-safe audit logging.

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
| **Qwen** | Orchestrator (default for this deployment) |
| **Luna** | Simple tasks, short answers, basic questions |
| **Spark** | Read-only code analysis, bounded coding subtasks |
| **Terra** | Durable default orchestrator, general-purpose tasks |
| **Sol** | Complex, security-sensitive, critical infrastructure |
| **Claude Opus 5** | Standalone diagnostic bridge (optional) |

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
  spark: false   # read-only, requires explicit [spark] label
  terra: true
  sol: true
  opus5: true
  qwen: true

# Default parent model
default_model: terra

# Delegation limits (in ~/.hermes/config.yaml)
delegation:
  max_concurrent_children: 2
  max_spawn_depth: 1
  max_iterations: 16

# Audit logging
logging:
  prompt_preview_chars: 240
  redact_prompt_preview: true

# Orchestration (auto fan-out)
orchestration:
  enabled: false
  max_tasks: 1
```

## Usage

### Explicit Model Override

Prefix your message with a tag:

```
[luna] Simple question
[spark] Read this file and report
[sol] Complex security analysis
[opus] Diagnostic review
```

### Delegation

The parent agent can delegate independent bounded subtasks:

```python
delegate_task(
  goal="Bounded independent subtask",
  context="Handoff capsule: objective, boundaries, relevant decisions, expected evidence. No user-facing output."
)
```

## Policy

1. **Stable Parent** — The user-facing conversation does not silently switch models.
2. **Delegation is an exception, not the default** — Only for genuinely independent subtasks.
3. **Worker limits enforced** — Max 2 concurrent children, 1 spawn depth, 16 iterations.
4. **Privacy-safe audit** — 240-char bounded preview, redacted sensitive data.
5. **Documented changes** — Every policy change updates README and tests.

## Tests

```bash
cd ~/.hermes/plugins/model-router
pytest test_*.py
```

## Version

**1.2.0** — Stable parent policy, bounded delegation, privacy-safe logging

## License

MIT

## Author

SENTINEL — Hermes Agent Model Router
