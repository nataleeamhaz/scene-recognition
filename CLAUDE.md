# CLAUDE.md

## Model selection for work in this repo

Match model capability to task complexity when spawning subagents or choosing
how to approach a task:

- **Simple, mechanical, well-scoped work** (e.g. a display-only UI tweak, a
  small isolated bug fix, boilerplate/CRUD-style changes, running/summarizing
  tests) — use a lower-cost model like Haiku.
- **More complex or judgment-heavy work** (e.g. architecture/design
  decisions, multi-file refactors, anything with real tradeoffs or ambiguity,
  security-sensitive changes) — use a more capable model (Sonnet/Opus).

When in doubt, or when a task has deciding factors that affect direction,
surface those to the user before implementing rather than picking silently.
