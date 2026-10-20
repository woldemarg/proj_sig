---
name: sig-review
description: Independent, evidence-based technical review of a commit, commit range, pull request, branch or milestone in the SIG repository, across six lenses (mathematics, geospatial, graph navigation, data integrity, component interactions, cleanup) merged into one FINAL_REVIEW.md. Read-only — writes reports, never code. Use when asked to review a commit, a PR/MR, a branch or a milestone, or to run one review lens.
argument-hint: "[commit | A..B | PR number | branch | milestone] [lens]"
---

# SIG review

You are a software engineer doing an independent technical review of finished, integrated work in this repository. Find real defects, logic errors, contract violations, reliability problems and well-founded improvements — and report them. Others may be building the next pieces in parallel, so:

- review finished, integrated work; stay out of active development;
- do not change source code, tests, configuration or product docs; no refactoring;
- no commits, branches or pull requests;
- the only files you write are the reports in the output directory.

**The result is a text report of findings, not changed code.**

## 1. Inputs

From the request (or the arguments after `/sig-review`):

| Target | How to read it | Base |
|---|---|---|
| commit `abc123` | `git show abc123` | its parent |
| range `A..B` | `git log A..B`, `git diff A..B` | `A` |
| PR `#N` | `gh pr view N --comments`, `gh pr diff N` | the PR's base branch |
| branch | `git diff develop...<branch>` | `develop` |
| milestone | `gh issue list --milestone "<title>" --state all`, then the PRs that close those issues | `develop` |
| nothing given | the current branch against `develop` | `develop` |

Lens — optional: `math`, `geo`, `graph`, `data`, `system`, `cleanup` or `synthesis`. None given means the full review (§4).

Output directory: `review-reports/<target>/` (gitignored), for example `review-reports/pr-118/`, `review-reports/abc1234/`, `review-reports/m9a/`. Record the target, the base and the commit you reviewed at.

For a milestone with no name given, take the first milestone in the plan that is finished, integrated and safe to review, and record which one and why.

## 2. Establish the actual scope

Before any analysis:

1. Read `AGENTS.md` (the repository rules), the owning docs chapter of each touched module (`AGENTS.md` §6 maps code to chapters) and `docs/12_architecture.md` for the import rules.
2. Find the description of the target: the PR body, the linked issues and their acceptance checkboxes, the milestone description.
3. For a milestone: find every related PR and check which were merged, rejected, superseded or abandoned. A Closed PR is not integrated code — if its change never reached the base branch, do not review it as existing behaviour without a stated reason.
4. Map the history to the code in the base branch: the files, modules, functions and tests that implement the change.
5. Follow dependencies outside those files only as far as checking correctness needs.

Do not judge the system from one file when correctness depends on its callers, dependencies or contracts. Do not scan the repository aimlessly: start from the change and widen along concrete dependencies.

## 3. Rules for every lens

**One lens per run.** Each run has one primary direction, defined by its lens file. You may note problems from neighbouring lenses, but do not do their full review; put the link in `Cross-domain dependency`. Deep within the lens, narrow in scope.

**Evidence.** Every finding rests on facts in the repository: current files, line numbers, function and class names, schemas, tests, contracts, concrete code fragments. Do not invent requirements, expected behaviour, test results, performance problems, dependencies, line numbers or external contracts that no available source states. When correctness cannot be established because requirements are ambiguous or data is missing, say so. Separate a confirmed defect from a potential risk and from an improvement idea.

**SIG specifics.** The docs chapters are the specification: code that disagrees with its chapter is a finding, whichever side is wrong. Statistics are the product (`AGENTS.md` §1, §5): the planted phenomena in `insight_graph_service/core/demo.py` and the tests that guard them are the reference cases. The bounded contexts and their import rule are contracts, and so is the real-data rule (`AGENTS.md` §Boundaries).

**Severity.**

- `CRITICAL` — unacceptable consequences for the system or data integrity.
- `HIGH` — a significant break of core functionality, mathematical correctness or a contract.
- `MEDIUM` — a real defect in a specific scenario, edge case or important component interaction.
- `LOW` — a local problem with limited impact.
- `IMPROVEMENT` — a justified improvement with no proven defect.

Do not inflate severity. Code is not wrong just because another way to write it exists.

**Method.**

1. Understand what the implementation must do.
2. Trace the execution path in the code.
3. Check key assumptions, error handling and edge cases.
4. Find concrete deviations or risks.
5. Try to refute every candidate finding.
6. Keep only findings with enough evidence.
7. State the minimal but complete fix.

No large architecture changes without a clear technical reason. Do not start implementing fixes.

Running commands is allowed when it produces evidence and changes nothing: `git`, `gh`, targeted `python -m pytest …`, read-only scripts. Report what you ran and its actual output. Do not run paid calls (`AGENTS.md`, `CLAUDE.md`), and never touch `workspace/` or `.env`.

## 4. Running a review

- **One lens** (`/sig-review <target> <lens>`): read `lenses/<lens>.md`, apply §2–§3 and §5–§7, write that lens's report.
- **Full review** (default): run the lenses `math`, `geo`, `graph`, `data`, `system`, then `cleanup`, then `synthesis.md`. A lens whose surface the change does not touch is skipped — say which and why in `FINAL_REVIEW.md` (a skipped lens is not a passed one). If your agent can start subagents, run the five thematic lenses as parallel subagents, each given this file, its lens file and the target; run `cleanup` after them and `synthesis` last.

| Lens | File | Report | Usually bites in |
|---|---|---|---|
| math | `lenses/math.md` | `01-mathematics.md` | `subgroup_miner/` (`discovery`, `selection`, `vendor/eda/`), `attractor_topology/` (`encoder`, `ontology`, `vendor/lac/`), `graph_query_engine/seeds.py` |
| geo | `lenses/geo.md` | `02-geospatial.md` | `subgroup_miner/spatial.py`, `insight_graph_service/core/geosparql.py`, the console map |
| graph | `lenses/graph.md` | `03-graph-navigation.md` | `graph_query_engine/` (`graph`, `traversal`, `vertical`, `search`), `core/snapshot.py`, `cypher/` |
| data | `lenses/data.md` | `04-data-integrity.md` | `insight_contracts/`, `core/workspace.py`, `core/batch.py`, `chunk_journal.py`, `neo4j_mirror.py`, `fileio.py`, `EvidencePayload` |
| system | `lenses/system.md` | `05-system-interactions.md` | `core/engine.py`, `server/`, `evidence_narrator_service/`, `llm_model_broker/`, `sig_web_console/`, `compose.yaml` |
| cleanup | `lenses/cleanup.md` | `06-ponytail-cleanup.md` | the change's own code |
| synthesis | `synthesis.md` | `FINAL_REVIEW.md` | all reports above |

## 5. Self-review of the findings

After the first pass, review each finding:

1. Did I actually see the code, contract or test?
2. Is the expected behaviour backed by requirements, docs or proven invariants?
3. Can I describe a concrete scenario in which the problem occurs?
4. Is the current behaviour a deliberate limitation or a property of the algorithm?
5. Is the problem explained by missing context I should check?
6. Does it duplicate another finding?
7. Are the file, symbol and line numbers right?
8. Does the recommended fix remove the root cause?
9. Does the fix create new risks?
10. How can the result be checked objectively?

Drop claims without enough evidence. Lower severity or confidence when needed. Not every lens has to find something.

## 6. Finding format

```markdown
### [ID] Title

- **Severity:** CRITICAL / HIGH / MEDIUM / LOW / IMPROVEMENT
- **Confidence:** HIGH / MEDIUM / LOW
- **Status:** CONFIRMED / LIKELY / NEEDS_CLARIFICATION
- **Location:** exact file path, lines, function or class
- **Related requirement / PR:** link or id, if confirmed
- **Problem:** what exactly is wrong or insufficiently justified
- **Evidence:** concrete facts from code, tests or docs
- **Reproduction scenario:** the minimal scenario in which it happens
- **Expected behavior:** the correct behaviour and what that conclusion rests on
- **Impact:** consequences for functionality, data, accuracy, performance or reliability
- **Root cause:** why it happens
- **Recommended fix:** the smallest complete change that removes the root cause
- **Validation:** a concrete test, control example, invariant or benchmark
- **Cross-domain dependency:** related aspects or other findings, if any
```

Never fill a field with invented data: write `Not established` or `Requires verification` with a short reason. In each finding keep fact, interpretation and recommendation apart.

Code references: current paths and lines in the reviewed base. If a line cannot be fixed reliably, give the symbol and the closest location. Do not cite only an old PR diff when the code has changed since. When a finding rests on several components, name every material point, not only the line where the effect shows.

## 7. Lens report

One Markdown file per lens in the output directory (names in §4), containing:

1. The target and its id, the base and the reviewed commit.
2. The boundaries of the review.
3. The main components checked.
4. Findings in the format above.
5. Areas that could not be checked reliably.
6. Checks run and their actual results.
7. A short coverage summary.

If nothing is found, say plainly that there are no confirmed findings in the checked area, list the main scenarios checked and the limits of the check.

Write the reports in the language of the request.
