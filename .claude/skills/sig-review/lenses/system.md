# Lens: end-to-end flow

Report: `05-system-interactions.md`. Apply the rules of `../SKILL.md`.

Review the scenario, not the parts. Walk the change from the first call to the last result a caller can see, across the services it touches. `docs/12_architecture.md` is the service and import contract; `docs/09_operations.md` is the batch and the error codes.

Run this lens after the additive lenses. Do not read their reports, and do not re-derive what they own: a formula (`math`), a place (`geo`), a legal move (`graph`), a stored field (`data`), a calendar (`temporal`), a screen (`ux`). When the scenario shows one of those is wrong, name it in `Cross-domain dependency` and review only the flow around it. Dead code is `cleanup`.

## What to check

### 1. The scenario

Follow one path the change actually adds or alters. Typical paths, when the change touches them:

- upload → batch stages → commit → a reader sees the graph;
- a question → the narrator → search → the evidence → the answer the client gets;
- a map zone or a period leaving the console and coming back on the evidence;
- a deletion, a `PATCH`, a reset, a degraded workspace.

Check:

- the call order the chapters state;
- the arguments the next service needs are the ones the previous produced;
- a failure, a timeout, a cancel or a refusal stops where `docs/09` says, and leaves a state the next read can explain;
- a second call that should not change the result does not;
- a newer result is not overwritten by an older in-flight one;
- two services do not both own the same step.

### 2. The fact at the end

Take one value the scenario claims to carry (a weight, a cell, a path, a period, a digest). Check that the result at the end is that same value, not a default, a previous request's, or a partial run's.

If it differs because the estimator, the geometry, the walk, the record or the calendar is wrong, the finding belongs to that lens. If it differs because the scenario dropped, reordered or raced it, the finding belongs here.

### 3. What is not a finding

A slow path with no stated bound and no change in the result. A layer of calls the chapter requires. A quantity that is wrong for reasons inside one function — write the cross-domain line and stop.

## Expected result

For each finding:

- the scenario, first call to last result;
- the services in order;
- the step where the flow breaks;
- what the caller receives instead;
- the smallest change in the flow;
- how to see it: the request, the status, and the result a later read returns.

Keep a broken scenario apart from a slow one.
