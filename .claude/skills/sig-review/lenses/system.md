# Lens: component interaction, system integrity and efficiency

Report: `05-system-interactions.md`. Apply the rules of `../SKILL.md`.

Review the interaction between the components of the target. Find problems invisible in a single function that appear at the seams: wrong operation order, inconsistent contracts, redundant calls, inefficient computation chains, wrong error handling. The services, their contracts and the import rule are in `docs/12_architecture.md`; error codes and batch stages in `docs/09_operations.md`.

## What to check

### 1. Component interaction

Follow the key scenarios from the first call to the final result.

- Do the interfaces match their actual use?
- Are the required arguments and context passed?
- Is the call order right?
- Is there a hidden dependency on earlier state?
- Are results and errors handled consistently?
- Do components duplicate each other's responsibility?
- Can intermediate and final state diverge?

### 2. Integrity of the execution scenario

Check sequences where several locally correct components together produce a wrong result:

- early termination;
- use of stale results;
- repeated side effects;
- inconsistent state updates;
- wrong error propagation;
- mismatch between start and finish conditions;
- repeated calls that should not change the result.

### 3. Performance of system interaction

Look for confirmed or well-founded problems:

- repeated computation of the same results;
- loading the same data several times;
- redundant database or API calls;
- excessive serialisation;
- inefficient processing order;
- passing excessive data between components;
- redundant passes over collections;
- duplicate requests within one scenario;
- excessive memory use;
- a missed, justified chance for batching.

Do not call something a performance problem from how the code looks. If the impact is not measured, explain the mechanism, the expected impact and the limits of the conclusion.

### 4. Reliability and recovery

Check that the system keeps a correct state when one component fails. Depending on the implementation:

- partial execution;
- retries;
- timeouts;
- cancellation;
- unexpected responses;
- resource reuse;
- concurrent execution;
- idempotency and recovery after failure.

### 5. Minimal sufficient complexity

Identify needless call layers, repeated data conversion, redundant components or complex interaction schemes with no practical benefit. Do not propose a simplification that hides the real problem or lowers reliability.

## Expected result

For each finding:

- the full relevant chain of components;
- the order of operations;
- the point where correctness breaks or the extra cost arises;
- the actual impact;
- the recommended change;
- the matching regression tests or way to measure.

Keep proven interaction problems apart from performance hypotheses.
