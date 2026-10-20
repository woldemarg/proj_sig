# Lens: cleanup (ponytail)

Report: `06-ponytail-cleanup.md`. Apply §1–§3 of `../SKILL.md`; this lens has its own self-review and report format below. Run it after the five thematic lenses.

## 1. Task

Search for safe cleanup and simplification of the target's code.

If a `ponytail` skill is available to you (for example `ponytail-review`), load it, read its current instructions and follow its analysis and validation procedure. If it is not available, say so in the report, apply the criteria below, and do not claim a ponytail review was run.

## 2. Scope

Only the code of the target and its necessary dependencies. Look for:

- dead or effectively unused code;
- repeated implementations of the same logic;
- duplication between components or PRs;
- redundant computations and conversions;
- needless intermediate structures;
- obsolete execution paths;
- needless abstractions;
- unneeded dependencies;
- redundant checks;
- code that makes maintenance harder with no practical benefit.

`AGENTS.md` §2 and §8 (the consolidation pass) are this repository's own cleanup rules.

## 3. When removal or simplification is safe

Code is not redundant just because it has no obvious direct callers. Before concluding, check whether it can be used through:

- the public API (the HTTP API, `insight_contracts`);
- external clients;
- dynamic registration;
- dependency injection;
- configuration (settings read from the environment);
- reflection or other dynamic access;
- integrations, plugins or serialisation (journal and snapshot records, the Neo4j mirror);
- tests or backward-compatibility contracts.

Do not propose removing code whose role is not established. Classify:

- **Confirmed dead code** — non-use is sufficiently confirmed;
- **Redundant implementation** — the duplication has concrete evidence;
- **Safe simplification** — keeps contracts and behaviour;
- **Potential improvement** — there is a reason to optimise, but more evidence is needed.

## 4. What good cleanup means

For each candidate explain why the code can be removed or simplified, what that improves and how to confirm behaviour is kept. No cosmetic refactoring just to cut lines.

Do not change code or tests, delete files, commit or open a PR.

## 5. Self-review

- Is the code really not needed?
- Are indirect dependencies missed?
- Does the simplification bring a practical benefit?
- Is public and internal behaviour kept?
- Can the safety of the change be verified?

Drop unconfirmed recommendations.

## 6. Report

For each finding:

- ID;
- severity;
- exact location in the code;
- the redundant or duplicated implementation;
- evidence;
- why it is not needed or what harm it does;
- the recommended simplification;
- dependencies to take into account;
- a test or other way to confirm the change is safe;
- confidence.

Where evidence is insufficient, do not recommend deletion; record the question that needs checking.
