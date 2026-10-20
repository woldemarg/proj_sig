# Final synthesis

Report: `FINAL_REVIEW.md`. Apply `SKILL.md`; the verification and report structure below replace its §5 and §7.

You are the reviewer who turns all the lens reports of the target into one final, consistent report. The intermediate reports, in the output directory:

- `01-mathematics.md`;
- `02-geospatial.md`;
- `03-graph-navigation.md`;
- `04-data-integrity.md`;
- `05-system-interactions.md`;
- `06-ponytail-cleanup.md`.

Find the files actually present. A missing report is not a passed check: name it, and the reason if the lens was skipped as not applicable.

Do not rewrite the reports mechanically; turn them into one technical document from which an engineer understands every material problem and can plan the fixes.

## 1. Verify the findings

For each finding:

1. Find the code at the reviewed revision.
2. Check that file, symbol and lines are current.
3. Confirm the description matches the code.
4. Check whether a commit after the reviewed revision (the branch tip now) has already fixed it.
5. Match the evidence against the claim and the proposed fix.
6. Check whether it repeats another finding.
7. Keep the uncertainty where the problem could not be confirmed.

Do not trust an intermediate report: it is a source of hypotheses to check. An unconfirmed finding does not go into the confirmed defects; move it to the uncertainties with a reason if needed.

## 2. Deduplicate and reconcile

Find findings that describe one root cause from different angles. Merge them only when they truly are the same defect, and then:

- keep all material evidence;
- list every affected component;
- explain how the symptoms relate;
- give one recommended fix;
- list the tests that cover every material aspect.

Do not merge different defects just because they are in one file.

If two findings contradict each other, re-check code, requirements and contracts. If the evidence cannot resolve it, record the contradiction.

Do not lower severity because one problem shows up in several lenses.

## 3. One remediation plan

For each confirmed finding give the best technical fix. It must:

- remove the root cause;
- be minimally sufficient;
- keep the required contracts;
- account for dependencies between components;
- not produce duplicate or conflicting changes;
- include a way to verify the result.

Determine the dependencies between fixes. If a mathematical error is the root cause of a graph-search problem, do not plan a fix of the symptom only, leaving the wrong computation. If a cleanup depends on a business-logic fix, state the order.

Do not implement the fixes.

## 4. Structure of `FINAL_REVIEW.md`

### 1. Executive summary

- the target reviewed;
- the components in scope;
- the key risks found;
- the overall state of technical correctness that can be justified.

No conclusion of full correctness if not every lens could be checked.

### 2. Scope and coverage

- the related issues and PRs established;
- the components actually checked;
- the lenses run;
- missing, skipped or incomplete checks;
- the key limits of the evidence.

### 3. Findings summary

| ID | Severity | Lens | Component | Short description | Confidence |
|---|---|---|---|---|---|

Only unique findings, after verification and deduplication.

### 4. Detailed findings

Each material finding in the §6 format of `SKILL.md`, with the required regression tests under `Validation` and the related findings and dependencies under `Cross-domain dependency`.

Detailed enough to hand the finding to another coding agent for fixing without repeating the review.

### 5. Cross-domain issues

Problems that run through several lenses. For example, a coordinate-conversion error can distort distances and then cause wrong graph navigation: explain the causal chain and name the root defect. Refer to finding IDs; do not repeat the full descriptions.

### 6. Recommended remediation order

An ordered fix plan that accounts for:

- severity;
- dependencies between problems;
- fixing the root cause first;
- regression risk;
- ease and safety of verification.

For each stage: which findings to fix and how to verify. A technically justified order, not a mechanical sort by severity.

### 7. Cleanup opportunities

From the ponytail review:

- confirmed redundant code;
- safe simplifications;
- duplication to remove;
- sufficiently justified optimisations;
- questions to check before removing anything.

Keep confirmed defects and optional improvements apart.

### 8. Tests and verification

The checks actually run and their results; separately, the unavailable or failed checks. Do not claim tests passed without confirmation. Each finding has clear next steps to verify its fix.

### 9. Unresolved questions and limitations

Only open questions that materially affect correctness, scope or the choice of fix. For each:

- what is unknown;
- why it matters;
- what information is needed;
- which decisions cannot be made without it.

### 10. Traceability

Link the final findings to the intermediate reports, code files, issues and original PRs where the ids are available. Keep the finding IDs or the links between the original and the merged findings.

## 5. Final check

Before saving, confirm that:

- every confirmed finding has evidence;
- the code locations are current;
- duplicates are removed;
- contradictions are not hidden;
- recommendations are concrete and actionable;
- root causes are separated from consequences;
- cleanup is separated from functional defects;
- test results match the facts;
- unchecked lenses are marked;
- the document stands on its own as a technical artifact.

Write only `FINAL_REVIEW.md`. Do not change code, tests, configuration or product docs. No PR.

**The result is one consistent, evidence-based, actionable report: what is wrong, where exactly, why it is a problem, how to fix it correctly and how to confirm the fix.**
