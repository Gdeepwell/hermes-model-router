# Claude Opus 5 coding tier — dispatch plan

**SUPERVISOR DECISION: ACCEPT.** The existing OpenAI-compatible request middleware must not rewrite an `openai-codex` request to an Anthropic model. Claude Code OAuth is therefore reached only through a separate, explicit Claude Code execution bridge.

## Contract

1. Preserve Luna/Spark/Terra/Sol routing unchanged for ordinary Hermes requests; preserve Sol-only design/CSS/UI policy.
2. Add an `opus5` **coding-agent tier** selected only by conservative coding classification or `[opus5]` / `[opus]` manual override. It is not a same-provider model rewrite.
3. The bridge uses `claude -p` with `--model opus`, normal Claude Code permission controls, `--max-turns 8`, and `--max-budget-usd 5`; it does not pass a permission bypass flag.
4. Resolve the canonical effective primary model from Claude Code's JSON `modelUsage`. For an `opus` review, accept when `claude-opus-5` is present even if smaller internal/delegated models (for example `claude-haiku-4-5`) also appear; reject only when `claude-opus-5` is absent. Append the canonical effective route to the existing JSONL log.
5. Add `opus5` to terminal summaries and every web-viewer surface: filter, card, CSS color, pills, execution-tree kind, grouped/raw summary counting.
6. Test non-rewrite routing, conservative/manual bridge selection, model-use validation, route-log contract, viewer grouped nested filtering, and raw counting.
7. Verify with a real read-only Claude Code run and inspect the fresh JSONL entry plus served viewer HTML/API after restart.
