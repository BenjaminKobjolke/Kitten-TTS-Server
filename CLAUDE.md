# Version
2

# Coding Rules (Pointer)

This project's rules live in two files in the project root:

- `CODING_RULES.md` — code quality and conventions. BINDING for all code work in
  this repository, always.
- `IMPLEMENTATION_FLOW.md` — the end-to-end flow to follow when planning and
  implementing a change (checks, gates, Definition of Done).

MANDATORY: Before writing or editing ANY code, you MUST Read BOTH files in full
**in the current session**. Do not rely on memory of a previous session, a
summary, or partial reads.

If you are about to make a code change and have not read both files in this
session: STOP, read them, then continue.

An external tool may orchestrate the implementation itself. When a skill or run
states that the implementation is orchestrated, ignore `IMPLEMENTATION_FLOW.md`
entirely — the orchestrator runs those steps as its own phases. `CODING_RULES.md`
stays binding either way.

Do not inline rules back into this file and do not use `@import` for either file —
they are intentionally referenced, not imported.


## Graphify scan scope

This project scans the repository root (`.`) so the top-level Python service modules and `ui/script.js` are included. `.graphifyignore` excludes documentation, settings, static assets, and other non-code inputs.

## Code Analysis

Use the changed-files analyzer after implementing a feature, finishing a plan, or fixing a bug:

```bash
powershell -Command "cd 'D:\GIT\BenjaminKobjolke\Kitten-TTS-Server'; cmd /c '.\tools\analyze_changed_and_new_files.bat'"
```

This runs cli-code-analyzer with `--only-changed`, so it reports findings only for new or modified files while project-wide analyzers still run. Use `tools/analyze_code.bat` for a full-project audit, exception maintenance, before a release or merge, after a shared-code refactor, or when the working tree is clean. Results are written to `code_analysis_results/` as per-rule CSV files; a missing report means that rule found no findings.
