# Lens: graph navigation

Report: `03-graph-navigation.md`. Apply the rules of `../SKILL.md`.

Review the graph-navigation logic of the target: does the system model the links between nodes, the moves between them, path search and the rules for reaching the wanted nodes or results correctly? `docs/06_graph_and_storage.md` and `docs/07_question_answering.md` define the graph and its traversal grammar.

Establish the actual graph model first. Do not assume it is directed, weighted, acyclic, or even backed by a graph library.

## What to check

### 1. Graph semantics

- What do nodes and edges represent?
- Are edge directions consistent?
- Are the types, properties and constraints of the links right?
- Do the actual moves match the graph's domain meaning?
- Is a real link replaced by mere similarity or another indirect sign without justification?

### 2. Navigation algorithm

For the algorithms actually present:

- choice of start nodes;
- which moves are allowed;
- node-visiting rules;
- tracking of the path already walked;
- termination conditions;
- depth or step limits;
- cycle handling;
- behaviour on disconnected graphs;
- choice among alternative paths;
- correctness of weights and move costs, if used.

### 3. Reachability and completeness

Check that the algorithm can reach the nodes it must find and does not drop correct paths through wrong filters, traversal order or early termination.

If the search is deliberately heuristic or bounded, judge it against its declared guarantees. Do not demand a full traversal where selective search is the design.

### 4. Correct results

- Do all edges of a result path exist?
- Are edge directions and move constraints respected?
- Is path length mixed up with weight, similarity or another metric?
- Are repeated nodes and edges handled correctly?
- Is the result stable under the same conditions, if that is expected?
- Is context from earlier moves, needed for the next decision, lost?

### 5. Integration with other components

Follow how traversal results reach the next stages. Check that information about the path, weight, link type, previous state or move source is not lost where later logic depends on it.

General serialisation and long-term storage belong to the `data` lens.

## Expected result

For each problem give a minimal graph or sequence of moves that shows the defect:

- the start nodes and relevant edges;
- the expected path or result;
- the actual behaviour;
- the condition or operation that causes the wrong result;
- the smallest complete fix;
- a regression test that tells correct from wrong behaviour.

Do not recommend replacing the whole algorithm just because another implementation exists. Find the concrete cause first.
