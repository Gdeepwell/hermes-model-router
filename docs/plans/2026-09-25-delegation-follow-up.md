# Hermes delegation: remaining implementation plan

Updated 2026-09-25. Status: all seven follow-up items implemented and verified.

Implementation commits: `dfea3b5` (account admission), `d6f7ad8` (retry
schema), `5ce8cbc` (CLI adjustment audit), `2d8902a` (conductor and workflow),
`4c80665` (bridge limits and lifecycle), `2488b7e` (settings failure paths),
and `3bd4a76` (activity snapshot cache).

All completed fixes are retained, including the lower-priority changes. The
scope-reduction reverts have been reversed. The application tree is identical
to the completed-fix tree at `f5f00c8`; this document is the new deliverable.

## Completed work to preserve

The six P1 changes cover guard-before-balancing, shared conductor worker order,
hard admission on both accounts, host-capability checks, current task-schema
contracts, and stable parent identity. Their original implementation commits are
`1aedbe3`, `8e47b87`, `b653043`, `afcca1b`, `37d09ae`, and `a82fa2f`.

The following later changes are also retained, rather than scheduled again:

- Label-independent retry classification and guarded retry ordering (#7).
- Stale/reset usage handling (#8) and tool-independent Claude recognition (#9).
- Visible-card filter refresh (#10) and activity routing reasons (#11).
- Selected-tier CLI switches/usage policy (#12).
- Bridge origin/limit forwarding, model identity, and terminal launch errors (#13).
- Validated, serialized, revision-checked settings saves with rollback (#14).
- Codex-only worker enforcement, explicit Claude fallback-policy copy, removal
  of superseded renderers and unused helpers, inert-setting cleanup, and the
  corrected conductor documentation.

The remaining work below addresses cross-feature interactions, missing
integration coverage, and the deferred activity-log performance finding.
Each implementation item should have its own commit and focused regression
coverage. Avoid bundling it with already-completed changes.

## 1. Admit the account that will actually execute a CLI worker

Priority: first. Follow-up to review #3/#12.

Files: `__init__.py` (`run_llm_with_transient_failover`, `_maybe_run_opus5`),
`worker_admission.py`, `test_worker_admission.py`, `test_bridge_policy.py`.

Remaining gap: execution middleware checks the incoming worker's provider before
attempting the Claude CLI bridge. A Codex placeholder request can therefore be
stopped by a Codex hard limit even though the eligible worker would execute on
a healthy Claude account. The newly shared guard is correct per account, but
its placement can reject the wrong execution account.

Implementation:

- Separate bridge eligibility/route selection from execution so the selected
  account is known before admission. Do not launch a subprocess during probing.
- For an eligible Claude bridge, enforce Claude's switch, cooldown, and usage
  guard before execution. Preserve the existing root/worker and read-only rules.
- If the bridge is ineligible or fails and the request returns to Codex, check
  Codex admission immediately before its provider call. A bridge failure must
  not bypass Codex's hard limit.
- Avoid Claude usage/auth probes on requests that cannot use the bridge.
- Preserve the host middleware's single-use `next_call` contract and explicit
  worker-stop response; exceptions before `next_call` can fail open in the host.

Acceptance: mock both transports and test Codex-closed/Claude-open,
Claude-closed/Codex-open, both closed, bridge failure with Codex closed, disabled
aliases, and ordinary roots. Assert exactly which transport ran and that closed
accounts received no calls.

## 2. Deliver usable retry instructions through the current host schema

Priority: after item 1. Follow-up to review #5/#7.

Files: `__init__.py` (`_quota_redispatch_instruction`, `_failed_delegation_blocks`,
`_dispatch_phrase`, `_chain_entries`), `test_quota_redispatch.py`.

Remaining gaps:

- The explicit `[ROUTER WORKER STOPPED]` response can arrive in a host envelope
  marked completed. Retry detection currently selects error/warning task blocks
  and generated error lines, so it can miss this deliberate admission stop.
- When the Claude tool is inactive, retry wording can still prescribe a `model:`
  parameter that the current `tasks[]` schema does not offer.
- The inactive retry chain still requires named host targets, even for local
  tiers reachable through goal prefixes.

Implementation:

- Recognize the router's explicit stop contract in an actual delegation result
  without treating arbitrary worker prose about quotas as a retry signal.
- Use schema-aware dispatch wording: local goal prefixes for the current host,
  a model parameter only when present, or `delegate_claude` when reachable.
- Filter retries by actual reachability and current shared guard/balance order.
  Keep a labelled task's semantic kind and preserve its committed progress.
- Distinguish known cooldown recovery from hard closure with unknown recovery;
  do not instruct retries that are guaranteed to fail at the host default route.

Acceptance: normalize retry calls with the installed host schema; cover explicit
stops delivered as completed results, ordinary task failures, quota-related task
text, local tiers absent from named targets, deferred Claude tools, and legacy
model-parameter hosts.

## 3. Complete CLI adjustment evidence and processing-UI integration coverage

Priority: after item 1. Follow-up to review #11/#12/#13.

Files: `__init__.py`, `claude_opus_bridge.py`, `agent_activity.py`,
`web_viewer.py`, the bridge/activity/dashboard tests.

Remaining gap: CLI soft-limit substitution changes the alias, but the shared
usage outcome is not carried through as requested-versus-effective tier evidence.
The activity-reason regression currently stops at the backend projection; it
uses a shortened reason that would not match the frontend's full step-down
pattern. The production reason propagation is fixed, but the whole display
path is not yet proven by that test.

Implementation:

- Carry requested tier, effective tier, and adjustment/refusal reason through
  CLI lifecycle/audit records, without fabricating successful model evidence.
- Use verified canonical output for the effective model; distinguish requested
  identity on a still-running process.
- Add a JSONL → activity API → rendered chip regression using actual router
  reason formatting and actual Claude audit/lifecycle fields.
- Cover nested workers, old records without reasons, skipped substitutions,
  refusal, and redaction. Avoid counting one bridge result as two workers.

Acceptance: equivalent real Codex and Claude adjustments produce consistent
indicators and hover evidence; skipped substitutions have no down arrow;
Sonnet does not acquire an Opus label.

## 4. Close workflow-policy gaps in conductor selection and live switching

Priority: after item 1. Follow-up to the review's workflow consistency finding.

Files: `__init__.py` (`_conductor_tier`, offering/eligibility helpers),
`worker_admission.py`, `test_workflow_switch.py`, conductor tests, README.

Remaining gap: Codex-only mode now rejects Claude workers and filters offered
Claude targets, but `_conductor_tier` separately selects an explicit callable
conductor. It should not recommend a route that admission will reject. Host
fallback behavior and live workflow changes also need a documented contract.

Implementation:

- Apply the same workflow eligibility predicate when selecting a conductor as
  when offering and admitting workers. Include custom host target names.
- Respect the current host's routing capabilities; an external conductor cannot
  be reached merely by putting its name in a goal prefix.
- Define the behavior of already-running Claude workers after a switch. Current
  execution admission stops their next call; document and test that behavior
  unless a deliberate policy change is chosen.
- Verify Claude reached through a Hermes-managed child fallback still passes
  execution admission, while a Claude parent and historical telemetry remain
  unaffected. Do not rewrite host provider configuration implicitly.

Acceptance: an explicit Claude conductor is not selected in Codex-only mode;
legacy/current hosts receive usable alternatives or a clear inability to
spawn; live-switch and fallback tests preserve parent identity and telemetry.

## 5. Finish bridge limit and lifecycle regression coverage

Priority: after item 3. Follow-up to review #13.

Files: `claude_opus_bridge.py`, adapter callers, bridge/activity tests.

- Validate `max_turns` as a positive integer before forming CLI arguments;
  currently a positive fraction can pass the numeric lower-bound check.
- Ensure a positive budget cannot round to `0.00` in argv. Choose either an
  explicit supported minimum or a precision-preserving representation.
- Add adapter-to-lifecycle tests with different parent and child session IDs;
  the current adapter test checks forwarding but does not prove host-context
  interpretation when both identifiers exist.
- Exercise subprocess-start `OSError` and verify a terminal error record and a
  non-running dashboard card. The handler is implemented; its end-to-end
  regression coverage remains to be added.

Acceptance: invalid numeric inputs launch nothing, valid limits reach argv
unchanged according to the documented contract, and launch failures never leave
a ghost-running child or attach it to the wrong parent.

## 6. Complete settings failure-path and UI test infrastructure coverage

Priority: independent of worker routing. Follow-up to review #10/#14.

The transactional save, frontend queue, conflict handling, and rollback are
implemented. Remaining validation work:

- Add browser/DOM tests for a failed first save with later edits queued: queued
  stale writes must stop, the error must remain visible, and a successful reload
  must restore the ability to save.
- Test a delayed settings GET racing with edits, and a stale revision from a
  second page. Do not silently replace newer local edits.
- Test rollback when an external writer changes the host file after the first
  transaction write; preserve the external change and report the conflict.
- Document the compatibility policy for clients that omit revision tokens.
  The dashboard always sends one; unversioned API clients currently bypass
  optimistic conflict rejection.
- Make the full-page DOM runner an explicit test/CI prerequisite. The current
  regression uses jsdom through `NODE_PATH` and skips when unavailable; CI should
  not silently omit the principal visible-filter test.

Keep the documented distinction between atomic replacement of each file and a
cross-file crash transaction. A recovery journal is optional future work, not a
requirement of the current ordinary-write-failure fix.

## 7. Avoid repeated full activity-log parsing

Priority: last; this was a scaling concern, not a measured production incident.

Files: `agent_activity.py` (`_router_calls_by_session`), dashboard endpoint
integration, activity tests.

- Measure repeated `/api/entries` and `/api/agents` refreshes first.
- Reuse a bounded parsed snapshot keyed by resolved file identity, size, and
  nanosecond modification time when the retained log has not changed.
- Invalidate on append, replacement, truncation, rotation, and disappearance.
  Ensure callers cannot mutate cached records used by another response.
- Bound memory and consider concurrent first reads; do not introduce a global
  cache that grows with every log version.
- Preserve the complete retained audit history. Do not replace complete scans
  with an arbitrary tail that drops older workers' route evidence.

Acceptance: unchanged refreshes reuse parsing, changes invalidate it, concurrent
responses remain consistent, and old retained workers keep their full history.

## Delivery and validation

Implement only when resumed with authorization for these remaining items. Use
one issue per commit, focused tests per change, and an integrated full-suite run
when the batch is complete. Use fresh temporary `HERMES_HOME` and
`CLAUDE_CONFIG_DIR`, strip credentials, mock providers/subprocesses, and bind HTTP
tests only to localhost. No paid model calls or live configuration changes are
needed.

Retained-fix validation: a fresh complete run on the restored application tree
passed all 737 tests, including localhost HTTP tests and the jsdom full-page
regression. `git diff --exit-code f5f00c8 HEAD` confirmed the restored application
tree exactly matched the completed-fix tree before this plan was committed.
