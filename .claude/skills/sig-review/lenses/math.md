# Lens: mathematics and algorithmic correctness

Report: `01-mathematics.md`. Apply the rules of `../SKILL.md`.

Review the mathematical and algorithmic correctness of the target: are the operations, algorithms, transformations, computational dependencies and decision criteria implemented correctly? The owning chapters (`docs/02`–`docs/05`, `docs/07`) and the formula index in `docs/11_reference.md` state the intended formulas.

## What to check

### 1. Formulas and mathematical semantics

- Do the formulas match the function's purpose?
- Are the operations, their order and their signs right?
- Are normalisation, aggregation, division and scaling applied correctly?
- Are units and conversions of quantities handled correctly?
- Are different mathematical semantics mixed in one computational chain?

### 2. Algorithm correctness

- Does the implementation match the described algorithm?
- Are termination conditions, selection criteria and state-update rules right?
- Are steps missing, or repeated in a way that changes the result?
- Are boundary values handled correctly?
- Are the required invariants kept throughout?

### 3. Numerical stability

Check the relevant cases:

- zero or very small denominators;
- empty data sets;
- `NaN`, `Infinity` and invalid values;
- overflow and loss of precision;
- rounding and floating-point comparison;
- error accumulation;
- unusual input sizes.

Do not hunt for all of these in every function — check them where the operations are actually present.

### 4. Boundary conditions and invariants

Identify the most important boundary scenarios. Check that the properties correctness relies on hold.

Use existing tests, independent control examples, reference calculations or algorithm properties where available — the planted phenomena of `insight_graph_service/core/demo.py` first.

Passing tests do not prove correctness.

### 5. Algorithmic efficiency

Note clearly inefficient operations, redundant recomputation and excessive complexity — only as far as they concern the mathematical algorithm. General component interaction belongs to the `system` lens.

## Expected result

Only the mathematical and algorithmic problems you could substantiate. For each:

- the problematic operation or formula;
- what its semantics should be;
- how the implementation deviates;
- the scenario that produces a wrong result;
- the minimal way to remove the root cause;
- a control example, invariant or test that verifies the fix.
