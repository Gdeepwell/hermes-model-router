# Model Router — Hermes Agent Plugin

Intelligent LLM routing for Hermes Agent. Routes between Luna (simple tasks), Spark (read-only coding), Terra (default orchestrator), Sol (complex/security-critical), and Claude Opus 5 bridge — with stable parent policy, bounded delegation, and privacy-safe audit logging.

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
| **Luna** | Simple tasks, short answers, basic questions |
| **Spark** | Read-only code analysis, bounded coding subtasks |
| **Terra** | Default orchestrator, general-purpose tasks |
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
- Max 1 spawn depth (flat hierarchy — no recursive spawning)
- Max 16 child iterations
- Handoff capsule required for every worker task

### Live Dashboard

```bash
python3 ~/.hermes/plugins/model-router/web_viewer.py
# Opens at http://localhost:8765
```

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
