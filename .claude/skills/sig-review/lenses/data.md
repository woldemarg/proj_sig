# Lens: data transfer, transformation and storage

Report: `04-data-integrity.md`. Apply the rules of `../SKILL.md`.

Review the integrity of information in the target: is data passed between components, transformed, stored, restored and used in later computations correctly? The contracts live in `insight_contracts/`; storage and the journal formats in `docs/06_graph_and_storage.md`; the batch record in `docs/09_operations.md`.

The core question: can the system silently lose, corrupt, mix up or misread information between two processing steps?

## What to check

### 1. Data-transfer contracts

- Do the expected types, fields and structures match between sender and receiver?
- Are all required fields passed?
- Are names, types, units and value semantics consistent?
- Is important metadata lost?
- Are missing, empty or invalid values handled correctly?
- Are there silent conversions that change the meaning of the data?

### 2. Chain of transformations

Follow a representative path from source to final consumer. For each material transformation:

- what came in;
- what changes;
- what must be preserved;
- what the next component receives;
- whether the actual result matches the contract.

Pay special attention to precision loss, dropped fields, wrong defaults, implicit type coercion and schema drift.

### 3. Storage and restoration

If the change uses a store, cache, files, a database or serialisation:

- is all required data written;
- can it be restored correctly after reading;
- does the meaning survive the round trip;
- are schema and format versions consistent;
- are types, precision, identifiers or metadata lost;
- are partial writes and read errors handled;
- is integrity between related records kept.

### 4. State consistency

Check for situations where components use incompatible versions of the same data. Consider transactions, update order, caching, concurrent changes, repeated calls and idempotency — only where relevant to the implementation.

### 5. Meaning of a stored value

Look for a structure that is formally right but refers to the wrong thing: mixed-up identifiers, a link to a record that was not the one written, a lost source, two versions of one record.

Whether the value is the right statistic, place, path, period or label is that lens. Whether the scenario called the writer and the reader in the wrong order is `system`. This lens owns the bytes between them.

## Expected result

For each finding build the concrete information path and show the point of loss or distortion:

- the data source;
- the intermediate transformations;
- the component that receives the result;
- which value or property is lost or changed;
- the consequence for the system;
- the smallest complete fix;
- an integrity, restoration or contract test.
