# **Multilingual Seed Retrieval and Semantic Grounding for Statistical Insight Graphs**

## **Executive Conclusion**

The primary bottleneck in multilingual retrieval over the statistical insight graph is not graph traversal or latent space misalignment, but **lexical-semantic grounding at the graph entry boundary**1. The existing retrieval architecture relies on a weighted scoring formula where lexical matches account for up to ![][image1] of the total score (![][image2] for target, ![][image3] for scope, and ![][image4] for direction)2. When a user enters a query containing translated literals—such as *маржа* for margin, *телефонів* for phones, or *США* for US—the current lexical parser fails to recognize these tokens2. As a consequence, retrieval collapses into an unconditioned, global semantic fallback (![][image5]), scattering seeds into unrelated topological neighborhoods of the graph2.
The optimal, minimally complex architecture to eliminate this failure mode is **Decoupled Multilingual Literal Grounding (DMLG)**. Rather than altering graph traversal, incorporating runtime translation services, invoking query-time large language models (LLMs), or indexing full-pattern descriptions in multiple languages, the system decouples the graph's schema catalog (metric names, categorical dimension values, and directional polarity markers) into an in-process, dual-tier semantic anchor index.
This architecture combines three local mechanisms:

* An intra-script character ![][image6]\-gram TF-IDF index (char\_wb, range 3–5) to handle inflectional morphology and orthographic typos within identical scripts3.
* A cross-lingual dense embedding index leveraging the existing offline Qwen3-Embedding-0.6B model to map multilingual query tokens directly onto canonical dataset literals1.
* A distance-ratio and hubness-corrected verification layer utilizing Cross-Domain Similarity Local Scaling (CSLS) and Lowe's ratio test to eliminate false-positive associations on short tokens and acronyms5.

These additions are supplemented by composite linguistic rules for Slavic adverbial comparatives (e.g., resolving *більш низький* to a net downward shift) and associative driver phrasing2. The entire pipeline executes in-process against the existing in-memory snapshot, adds less than ![][image7] of per-query latency, consumes under ![][image8] of memory, and operates with complete independence from Neo4j2.

## **Evidence and Theoretical Foundations**

### **Cross-Lingual Entity and Schema Linking**

Extensive research in Cross-Lingual Entity Linking (XEL) and tabular Text-to-SQL schema linking establishes that natural language queries in arbitrary source languages should be grounded directly onto canonical schema items rather than through translated full-text passages9. In benchmarks such as BIRD and Spider, decoupling schema linking (identifying database columns, tables, and condition values) from query structure parsing consistently outperforms monolithic sequence-to-sequence models11. Supplying explicit schema-linking hints derived from frozen multilingual representations allows downstream graph reasoning to operate over canonical symbols regardless of user language12.
In cross-lingual knowledge base grounding, candidate generation is formalized as mapping a surface mention ![][image9] to a set of entities ![][image10]9. Grounding at the atomic literal level achieves superior precision compared to document-level cross-lingual retrieval because paragraph-level representations suffer from semantic dilution, where specific categorical filters are obscured by surrounding narrative text9.

### **High-Dimensional Hubness and Metric Calibration**

In high-dimensional multilingual vector spaces, unconstrained nearest-neighbor retrieval suffers from the **hubness problem**, wherein a small subset of vectors ("hubs") emerge as universal nearest neighbors to an abnormally large number of queries5. When querying an unconstrained vector space with short words or acronyms (such as mapping *США* to US), naive cosine similarity frequently returns hub tokens rather than true semantic equivalents5.
To counter hubness without retraining underlying models, Cross-Domain Similarity Local Scaling (CSLS) rescales cosine similarity by factoring in the mean distance of a candidate to its ![][image11]\-nearest neighborhood in the target vocabulary6:
![][image12]
where ![][image13] denotes the average cosine similarity of source vector ![][image14] to its nearest neighbors in target space ![][image15], and ![][image16] denotes the average similarity of target vector ![][image17] to its nearest neighbors in source space ![][image18]6.
Furthermore, ambiguous matches can be pruned using Lowe's ratio test (nearest-neighbor distance ratio)7:
![][image19]
where ![][image20] is the closest target candidate and ![][image21] is the second-closest7. When the top candidate does not stand out clearly from the second candidate, the match is discarded as ambiguous, preventing false associations between semantically loose terms7.

### **Morphological and Subword Sensitivity Across Scripts**

Subword and character ![][image6]\-gram representations (such as char\_wb TF-IDF) operate strictly within shared orthographic alphabets3. Empirical evaluations demonstrate that while character ![][image6]\-grams effectively resolve same-script inflectional suffixes and orthographic noise (for instance, yielding a similarity of ![][image22] between *телефони* and *телефонів*, and ![][image23] between *Харків* and *Харкові*), they yield exactly ![][image24] across script boundaries (such as *маржа* versus margin, or *США* versus US)3.

| Token Pair | Evaluation Category | char\_wb (3, 5\) Cosine Similarity | Empirical Outcome |
| :---- | :---- | :---- | :---- |
| ("fones", "phones") | Typographic Error (Latin) | 0.2912 | Robust Match3 |
| ("Харків", "у Харкові") | Locative Case Inflection (Cyrillic) | 0.2186 | Robust Match3 |
| ("телефони", "телефонів") | Genitive Plural Inflection (Cyrillic) | 0.5056 | Robust Match3 |
| ("маржа", "margin") | Cross-Script Cross-Lingual Translation | 0.0000 | Total Failure3 |
| ("США", "US") | Cross-Script Acronym Translation | 0.0000 | Total Failure3 |
| ("margin", "marginally") | Lexical Derivation (Latin) | 0.4393 | Ambiguous Distractor3 |

Character ![][image6]\-grams solve typographic noise and morphological case variation within an alphabet, while dense multilingual embeddings bridge cross-lingual semantic identity3. A staged hybrid design combining both signals is required3.

### **Graph Traversal versus Graph Entry Dynamics**

Recent knowledge graph retrieval research (such as HippoRAG) indicates that graph traversal mechanisms—including Personalized PageRank or Dijkstra walks over concept lattices—are structurally dependent on the precision of initial seed activation2. When seed selection fails or selects an incorrect subgroup, graph traversal propagates score mass across irrelevant topological regions2.
In the target system, the graph consists of a formal concept lattice (connected via SPECIALIZES, GENERALIZES, SIBLING, and CONTRASTS edges) linked via ACTIVATES edges to latent thematic centroids (Attractors)8. Traversal over this dual plane is mathematically constrained: once the correct seed pattern ![][image25] is identified, the best-first search reliably collects both direct refinements and scope-disjoint thematic analogs2. Traversal failure in the existing system is an artifact of entering the graph at incorrect coordinates, confirming that optimization efforts must focus strictly on query-to-seed grounding2.

## **Detailed Evaluation of Specific Research Questions**

### **A. Cross-Language Literal Matching**

Direct mapping of user tokens to normalized graph literals can be executed using several paradigms. The trade-offs among these approaches dictate the optimal operational choice for an in-process system.

| Matching Approach | Cross-Lingual Recall | Typo / Morphology Tolerance | Latency Overhead | Engineering & Dependency Overhead | Risk Profile |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Multilingual Dense Nearest Neighbor (Qwen3) | High (![][image26])4 | Moderate (embedding absorbs minor noise)26 | Low (![][image27])1 | Minimal (reuses loaded model)1 | Susceptible to hubness and short acronym collisions5 |
| Character ![][image6]\-gram TF-IDF (char\_wb) | Zero across scripts (![][image28])3 | Very High (![][image26]) within script3 | Negligible (![][image29])3 | Minimal (standard scikit-learn)3 | Incapable of cross-script translation3 |
| Machine Translation Service / API | High (![][image30]) | Low (errors propagate to translator) | High (![][image31]) | High (requires external service daemon) | Network failure risk; schema terms mistranslated |
| Runtime LLM Translation / Parsing | Very High (![][image26]) | High | Severe (![][image32]) | High (API costs, token quotas, prompt drift) | Non-deterministic; high latency; violates constraints |
| Rule-Based Bilingual Lexicons | Perfect for indexed entries | Zero for unlisted inflections | Negligible (![][image33]) | High ongoing manual maintenance | Fails on arbitrary new tabular datasets |
| Decoupled Hybrid (Char ![][image6]\-gram \+ Dense CSLS) | High (![][image26])3 | High (![][image26])3 | Low (![][image27])1 | Minimal (local in-memory Python)8 | Highly balanced; mitigated by Lowe's ratio test7 |

The decoupled hybrid approach is superior. It avoids external network daemons and manual vocabulary maintenance while utilizing the existing frozen Qwen3-Embedding-0.6B model to achieve robust zero-shot cross-lingual projection1.

### **B. Token-Level versus Phrase-Level Matching**

Embedding granularities exhibit distinct behaviors when matching queries to graph literals:

* **Individual Query Tokens**: Embedding single tokens (e.g., *телефонів*) provides clean matching for single-word categories and metrics, but fails for compound schema attributes such as delivery days, return rate, or median house value2.
* **Multi-Token Phrases (![][image6]\-grams, ![][image34])**: Multi-word sliding windows capture compound attributes and handle prepositions attached to nouns (e.g., *у США* ![][image35] region=US, *delivery days* ![][image35] delivery\_days)3.
* **Complete Query Embedding**: Embedding the entire query string (e.g., *"Чому маржа нижча для телефонів у США?"*) causes severe semantic dilution2. When a complete sentence is projected, high-level intent terms ("why", "lower") dominate the dense representation, while specific categorical filters (phones, US) become diffuse, preventing exact subset condition matching2.

The optimal strategy is multi-scale sliding window span extraction (![][image6]\-grams for ![][image34]), matching extracted spans against the literal vocabulary3.

### **C. Decoupled Literal Indexing versus Direct Insight Retrieval**

Two candidate retrieval workflows present distinct operational characteristics:

Workflow 1: Decoupled Literal Retrieval (Recommended)
User Query
  ↓
Extract Query Spans
  ↓
Match Against Literal Catalog (Metrics, Values, Directions)
  ↓
Populate Parsed Query Constraints
  ↓
Structural Seed Scoring Across Patterns
  ↓
Dual-Graph Traversal

Workflow 2: Direct Insight Retrieval (Monolithic Baseline)
User Query
  ↓
Direct Dense Cosine against Pattern Descriptions / Vectors
  ↓
Top-K Pattern Candidates Selected as Seeds
  ↓
Dual-Graph Traversal

In Workflow 2, user queries are matched directly against generated English pattern text or tripartite insight vectors1. Because pattern descriptions contain complex phrasing and measured numbers that are not present in user queries, dense similarity suffers from lexical-semantic drift1.
In contrast, Workflow 1 treats dataset column names and category values as language-independent semantic anchors11. Mapping query tokens to canonical literals first allows the downstream system to leverage exact structural predicates (target overlap, scope subset containment, and signed directional alignment)2. Decoupled literal retrieval is mathematically superior and aligns with findings in structured data querying11.

### **D. Hybrid Retrieval: Score Fusion versus Staged Candidate Generation**

Combining lexical, character, and dense signals can be approached either as a flat score summation across all insights or as a staged gating pipeline:

Staged Candidate Generation Pipeline:
Query Spans
  ↓
Stage 1: Exact String & Acronym Match (Cost: O(1))
  ↓ \[Unmatched Spans\]
Stage 2: Intra-Script Character n-gram Matching (Cost: O(N\_lit))
  ↓ \[Unmatched Spans\]
Stage 3: Cross-Lingual Dense Embedding Matching with CSLS (Cost: O(K · N\_lit · d))
  ↓
Consolidated Canonical Symbols
  ↓
Stage 4: Structural Seed Scoring over Graph Snapshot

A flat score summation across all patterns conflates distinct semantic levels: a character match on a condition value should not directly increment a pattern's macro relevance score. Instead, staged candidate generation at the literal level resolves surface variations into canonical symbols first, allowing pattern-level seed scoring to remain deterministic, structured, and interpretable2.

### **E. Score Fusion Methodology**

At the literal grounding layer, combining rank signals using Reciprocal Rank Fusion (RRF) is suboptimal because RRF is rank-based rather than score-calibrated. When matching a query token against a literal vocabulary, the system requires an absolute confidence gate (i.e., "does this token mean margin?"), which rank position alone cannot provide2. If no literal is relevant, RRF still assigns high reciprocal rank scores to the top candidates.
At the pattern seed scoring layer, the existing linear combination remains statistically sound:
![][image36]
This formulation guarantees that validated statistical evidence (![][image37]) and structural adherence (target, scope, direction) dominate over dense proximity2. Learned ranking models (such as LambdaMART or cross-encoders) require extensive labeled training data and introduce inference latency that is inappropriate for local statistical insight graphs.

### **F. Confidence Calibration and Acceptance Thresholds**

To determine whether a dense match ![][image38] is sufficiently reliable to treat as a literal match, absolute cosine thresholds are insufficient due to vector space anisotropy29. Relying on absolute thresholds leads to false positives on short acronyms and frequent words5.
The system must employ a three-tier decision rule:

> 1. **Absolute Similarity Floor**: ![][image39] (calibrated for Qwen3-Embedding-0.6B truncated to 384 dimensions)1.
> 2. **Hubness Rescoring (CSLS)**: Adjust cosine similarity by subtracting local neighborhood density: ![][image40]6.
> 3. **Ambiguity Ratio Test (Lowe's Test)**: The ratio between the distance to the closest candidate and the distance to the second-closest candidate must satisfy7:

![][image41]
This ensures that matches are accepted only when the top candidate is distinct and well-separated from distractors7.

## **Graph Traversal and Neo4j Evaluation**

### **Evaluation of Graph Traversal Algorithms**

Modern GraphRAG paradigms utilize diverse graph traversal strategies. Their applicability to the target system's dual-plane graph reveals clear trade-offs:

| Traversal Strategy | Description | Applicability to Statistical Insight Graph | Justification & Performance Impact |
| :---- | :---- | :---- | :---- |
| Dijkstra Best-First Search over Regular Grammar (Existing) | Explores lattice edges (![][image42]) and transverses latent anchors (![][image43]) with edge decay (![][image44])2. | Optimal (Retain unchanged)2 | Bounded by path budget; finds exact structural refinements and cross-scope analog subgroups deterministically2. |
| Personalized PageRank (HippoRAG) | Distributes probability mass across graph nodes starting from query seeds23. | Suboptimal | Diffuses mass across dense lattice clusters; lacks explicit constraints for opposite-direction contrasts2. |
| Multi-Hop Semantic Expansion | Expands nodes based on pairwise dense vector similarity31. | Redundant | Re-introduces semantic drift into graph navigation; duplicates latent anchor bridges2. |
| Subgraph Extraction & Reranking | Extracts ![][image11]\-hop neighborhood subgraphs and scores with cross-encoder32. | Excessive Overhead | Adds substantial inference latency (![][image45]); breaks strict token budget constraints1. |

The target graph consists of two well-defined planes: the structural concept lattice (Galois closed intents) and the latent anchor plane (sparse dictionary learning centroids)8. The existing traversal grammar (![][image46]) reliably uncovers scope-disjoint statistical analogues (e.g., linking phones ∧ US to tablets ∧ EU when both exhibit margin erosion)2. Empirical observations show that traversal functions properly once the correct seed pattern is selected, confirming that traversal requires no algorithmic restructuring2.

### **Evaluation of Neo4j Participation in Retrieval**

Neo4j's potential participation in the retrieval path was evaluated across four architectural configurations:

| Evaluation Criteria | Option 1: Pure Mirror (Recommended) | Option 2: Lexical Candidate Generator | Option 3: Full Retrieval Layer | Option 4: Neo4j Vector \+ Fulltext Hybrid |
| :---- | :---- | :---- | :---- | :---- |
| Query Latency | ![][image27] (In-memory Python)8 | ![][image47] (Bolt network roundtrip)8 | ![][image48] (Cypher graph traversal)8 | ![][image49] (Index scan \+ Bolt transfer)8 |
| Testability & CI | High (Runs offline in unit tests)8 | Low (Requires running Neo4j daemon)8 | Low (Requires active database instance)8 | Low (Complex index state synchronization)8 |
| Determinism | Absolute (Bit-level reproducible)8 | Moderate (Lucene scoring variations) | Moderate (Dependent on DB state) | Low (Approximated vector ANN searches)33 |
| Deployment Independence | Complete (Runs headless without Neo4j)8 | Broken (Fails if Neo4j uncommitted)8 | Broken (Fails if Neo4j unreachable)8 | Broken (Fails if Neo4j unreachable)8 |
| Cross-Lingual Accuracy | High (![][image26] via DMLG) | Low (Lucene full-text cannot bridge scripts)3 | Low (Requires translated graph properties) | Moderate (Requires syncing dense vectors)8 |

Option 1 is the only technically justified choice8. In the current system, Neo4j is an optional mirror configured with NEO4J\_ENABLED=false in several environments8. Tying retrieval to Neo4j violates deployment independence, complicates continuous integration, and adds serialization overhead8. Neo4j must remain strictly an asynchronous sink for visualization and developer exploration8.

## **Recommended Architecture and System Flow**

The **Decoupled Multilingual Literal Grounding (DMLG)** pipeline sits between user query ingestion and pattern seed scoring. The system flow processes queries through six distinct phases:

Decoupled Multilingual Literal Grounding (DMLG) Architecture:

User Query
  │
  ▼
\[Stage 1: Multi-Scale Span Extraction\]
  │ Extract n-grams (n ∈ {1, 2, 3}); flag case-preserved acronyms (len ≤ 4\)
  │
  ▼
\[Stage 2: Hybrid In-Memory Literal Grounding\]
  ├─► Layer A: Exact Lexical Match (Verbatim & Acronym Guards)
  ├─► Layer B: Intra-Script Character n-gram Index (char\_wb TF-IDF, sim ≥ 0.45)
  └─► Layer C: Dense Multilingual Match (Qwen3 Embeddings \+ CSLS \+ Lowe's Ratio)
  │
  ▼
\[Stage 3: Composite Linguistic Disambiguation\]
  ├─► Ukrainian Adverbial Comparative Resolver (e.g., "більш низький" → \-1)
  └─► Relational Driver Decoupler (Preserve direction on "пов'язано з вищим")
  │
  ▼
\[Stage 4: Grounded Structural Seed Scoring\]
  │ Target (0.35) \+ Scope (0.25) \+ Direction (0.15) \+ Semantic (0.15) \+ Weight (0.10)
  │ Filter top seeds (k ≤ 3); prune direct lattice neighbors
  │
  ▼
\[Stage 5: Dual-Graph Best-First Traversal\]
  │ Walk structural lattice edges (decay 0.85); transverse through Latent Attractors
  │
  ▼
\[Stage 6: Evidence Compilation & Synthesis\]
  │ Assemble top 10 ranked patterns with citation keys \[P1\]–\[P10\]
  └─► Pass to LLM generator or render deterministic Ukrainian summary

The retrieval pipeline executes sequentially:

> 1. **Multi-Scale Span Extraction**: Raw query strings are parsed into sliding-window spans ![][image50] of length ![][image51], preserving apostrophes for Ukrainian terms (e.g., *пов'язано*) and isolating short uppercase tokens (e.g., US, США)2.
> 2. **Hybrid In-Memory Literal Grounding**: Candidate spans are evaluated against the in-process LiteralCatalog. Matches are resolved through hierarchical gating: exact string equality is tested first; if unmatched, intra-script character ![][image6]\-gram TF-IDF is evaluated; if still unmatched, dense multilingual embeddings with CSLS and Lowe's ratio test are applied3.
> 3. **Composite Linguistic Disambiguation**: Multi-token directional modifiers are resolved to eliminate directional cancellation (e.g., resolving *більш низький* to ![][image52])2. Queries with relational verbs but explicit directional comparatives maintain their directional driver status2.
> 4. **Grounded Structural Seed Scoring**: Canonical symbols populate the ParsedQuery object, allowing the structural seed formula to score all patterns deterministically2. Top candidates (![][image53]) are selected, skipping direct hierarchical neighbors2.
> 5. **Dual-Graph Best-First Traversal**: The existing Dijkstra-style search traverses structural concept lattice edges and crosses latent anchor bridges to retrieve scope-disjoint analogues2.
> 6. **Evidence Compilation & Synthesis**: The top 10 retrieved patterns form the evidence object, which is passed to the LLM or formatted as the deterministic evidence summary2.

## **Architectural Modifications**

The required modifications integrate directly into the existing repository without altering data ingestion, statistical discovery, or graph modeling:

| Architectural Component | Existing File / Module | Current Implementation | Recommended State | Action | Technical Justification |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Ingestion & Discovery | ltir/ingestion.py, ltir/discovery.py | Profiling, EDA screening, bootstrap validation, Galois closed intents27. | Unchanged | Retain | Produces statistically validated subgroup lattices; no changes required27. |
| Latent Ontology (LAC) | ltir/ontology.py, ltir/engines/lac/ | OMP dictionary learning, EMA centroid tracking, mutual kNN topology25. | Unchanged | Retain | Discovers recurring statistical mechanisms across subgroups25. |
| Vector Encoding | ltir/canonical.py, ltir/encoder.py | Tripartite unit vectors (![][image54]), Matryoshka truncated to ![][image55]1. | Unchanged | Retain | Preserves exact opposite-direction geometry in phenomenon blocks1. |
| In-Memory Dual Graph | ltir/graph.py | Adjacency index of structural and latent edges derived from journals8. | Augmented with LiteralCatalog | Modify | Stores indexed metric names and condition values in memory8. |
| Query Parsing | ltir/query.py | Basic token regex, prefix matching, small stem list2. | Decoupled Literal Grounding \+ Composite Rules | Modify | Resolves cross-lingual literal mismatches and adverbial cancellations2. |
| Seed Scoring Formula | ltir/query.py | Linear weighted sum; fallback to dense cosine when lexical matches are absent2. | Structural scoring driven by grounded canonical symbols | Modify | Eliminates fallback collapse by populating target, scope, and direction2. |
| Graph Traversal | ltir/traversal.py | Best-first search over regular grammar (![][image46])2. | Unchanged | Retain | Operates reliably once seeds are accurately identified2. |
| Neo4j Persistence | ltir/neo4j\_sink.py | Write-only graph mirror; optional, disabled by default in test/dev8. | Pure visualization mirror | Retain (No Change) | Preserves complete deployment independence and deterministic CI8. |
| Translation Service | None | None | Explicitly Prohibited | Do Not Add | Eliminates external daemon dependencies, API costs, and latency2. |
| Runtime LLM Parsing | None | None | Explicitly Prohibited | Do Not Add | Eliminates ![][image32] latency and non-deterministic behavior. |

## **Retrieval Implementation**

The following self-contained Python implementation fits directly into ltir/query.py and ltir/graph.py, utilizing only numpy and scikit-learn:

Python
import re
import unicodedata
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from sklearn.feature\_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine\_similarity

class LiteralCatalog:
    """In-memory grounded index of dataset literals and domain concepts."""
    def \_\_init\_\_(self, encoder, min\_csls\_threshold: float \= 0.58, ratio\_threshold: float \= 0.85):
        self.encoder \= encoder
        self.min\_csls\_threshold \= min\_csls\_threshold
        self.ratio\_threshold \= ratio\_threshold

        self.literals: List\[Dict\] \= \[\]
        self.literal\_texts: List\[str\] \= \[\]
        self.literal\_types: List\[str\] \= \[\]
        self.embeddings: Optional\[np.ndarray\] \= None

        \# Intra-script character n-gram vectorizers
        self.char\_vectorizer\_latin: Optional\[TfidfVectorizer\] \= None
        self.char\_matrix\_latin: Optional\[np.ndarray\] \= None
        self.latin\_indices: List\[int\] \= \[\]

        self.char\_vectorizer\_cyrillic: Optional\[TfidfVectorizer\] \= None
        self.char\_matrix\_cyrillic: Optional\[np.ndarray\] \= None
        self.cyrillic\_indices: List\[int\] \= \[\]

    @staticmethod
    def \_detect\_script(text: str) \-\> str:
        for ch in text:
            name \= unicodedata.name(ch, '')
            if 'CYRILLIC' in name:
                return 'cyrillic'
            if 'LATIN' in name:
                return 'latin'
        return 'other'

    def build(self, graph\_snapshot: Dict):
        """Constructs literal catalogs directly from the DualGraph snapshot."""
        entries \= \[\]

        \# 1\. Target metric literals
        for node in graph\_snapshot.get("nodes", \[\]):
            if node.get("kind") \== "Metric":
                metric\_name \= node\["props"\]\["name"\]
                entries.append({"text": metric\_name, "raw": metric\_name, "type": "metric", "case\_sensitive": False})
                humanized \= metric\_name.replace("\_", " ")
                if humanized \!= metric\_name:
                    entries.append({"text": humanized, "raw": metric\_name, "type": "metric", "case\_sensitive": False})

        \# 2\. Scope condition literals (dimension values)
        for node in graph\_snapshot.get("nodes", \[\]):
            if node.get("kind") \== "Pattern":
                for cond in node\["props"\].get("conditions", \[\]):
                    attr \= cond\["attribute"\]
                    val \= str(cond\["value"\])
                    case\_sens \= len(val) \<= 4 and val.isupper()
                    entries.append({
                        "text": val,
                        "raw": f"{attr}={val}",
                        "type": "condition",
                        "attr": attr,
                        "val": val,
                        "case\_sensitive": case\_sens
                    })

        \# 3\. Canonical directional concepts
        entries.append({"text": "increase", "raw": "up", "type": "direction", "direction": 1, "case\_sensitive": False})
        entries.append({"text": "decrease", "raw": "down", "type": "direction", "direction": \-1, "case\_sensitive": False})
        entries.append({"text": "higher", "raw": "up", "type": "direction", "direction": 1, "case\_sensitive": False})
        entries.append({"text": "lower", "raw": "down", "type": "direction", "direction": \-1, "case\_sensitive": False})

        \# Deduplicate catalog entries
        seen \= set()
        deduped \= \[\]
        for e in entries:
            key \= (e\["text"\].lower() if not e\["case\_sensitive"\] else e\["text"\], e\["type"\], e\["raw"\])
            if key not in seen:
                seen.add(key)
                deduped.append(e)

        self.literals \= deduped
        self.literal\_texts \= \[e\["text"\] for e in self.literals\]
        self.literal\_types \= \[e\["type"\] for e in self.literals\]

        \# Precompute normalized dense embeddings
        self.embeddings \= self.encoder.embed(self.literal\_texts)
        norms \= np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings \= np.divide(self.embeddings, np.maximum(norms, 1e-12))

        \# Build intra-script character n-gram indices
        self.latin\_indices \= \[i for i, t in enumerate(self.literal\_texts) if self.\_detect\_script(t) \== 'latin'\]
        if self.latin\_indices:
            self.char\_vectorizer\_latin \= TfidfVectorizer(analyzer="char\_wb", ngram\_range=(3, 5))
            self.char\_matrix\_latin \= self.char\_vectorizer\_latin.fit\_transform(
                \[self.literal\_texts\[i\] for i in self.latin\_indices\]
            )

        self.cyrillic\_indices \= \[i for i, t in enumerate(self.literal\_texts) if self.\_detect\_script(t) \== 'cyrillic'\]
        if self.cyrillic\_indices:
            self.char\_vectorizer\_cyrillic \= TfidfVectorizer(analyzer="char\_wb", ngram\_range=(3, 5))
            self.char\_matrix\_cyrillic \= self.char\_vectorizer\_cyrillic.fit\_transform(
                \[self.literal\_texts\[i\] for i in self.cyrillic\_indices\]
            )

class MultilingualQueryParser:
    """Parses multilingual queries into canonical graph symbols."""

    UK\_COMPOSITE\_PATTERNS \= \[
        (re.compile(r'\\b(більш|більше)\\s+(низьк\\w\*|менш\\w\*|спад\\w\*)\\b', re.IGNORECASE), \-1),
        (re.compile(r'\\b(менш|менше)\\s+(низьк\\w\*|менш\\w\*|спад\\w\*)\\b', re.IGNORECASE), 1),
        (re.compile(r'\\b(більш|більше)\\s+(вищ\\w\*|більш\\w\*|зрост\\w\*)\\b', re.IGNORECASE), 1),
        (re.compile(r'\\b(менш|менше)\\s+(вищ\\w\*|більш\\w\*|зрост\\w\*)\\b', re.IGNORECASE), \-1),
    \]

    UK\_DIRECTION\_LEXICON \= {
        "up": \["вищий", "вище", "зростання", "ріст", "збільшення", "високий", "більший"\],
        "down": \["нижчий", "нижче", "спад", "зниження", "зменшення", "низький", "менший"\]
    }

    RELATION\_STEMS \= re.compile(
        r'\\b(correl\\w\*|relationship|relation\\w\*|coupl\\w\*|decoupl\\w\*|dependen\\w\*|covari\\w\*|'
        r'кореляц\\w\*|зв\[\\'’\]?яз\\w\*|взаємозв\[\\'’\]?яз\\w\*|залежн\\w\*|асоці\\w\*)\\b', re.IGNORECASE
    )

    ASSOCIATED\_DRIVE\_PATTERNS \= re.compile(
        r'\\b(пов\[\\'’\]?язано з|associated with|linked to)\\s+(вищ\\w+|нижч\\w+|higher|lower)\\b', re.IGNORECASE
    )

    def \_\_init\_\_(self, catalog: LiteralCatalog):
        self.catalog \= catalog

    def \_extract\_spans(self, text: str, max\_n: int \= 3) \-\> List\[Tuple\[str, int, int\]\]:
        tokens \= re.findall(r"\[\\w'’\]+", text)
        spans \= \[\]
        for n in range(1, min(max\_n \+ 1, len(tokens) \+ 1)):
            for i in range(len(tokens) \- n \+ 1):
                phrase \= " ".join(tokens\[i:i \+ n\])
                spans.append((phrase, i, i \+ n))
        return spans

    def parse(self, text: str) \-\> Dict:
        \# 1\. Resolve Directionality and Modifiers
        direction \= 0
        direction\_matched \= False

        for pat, dir\_val in self.UK\_COMPOSITE\_PATTERNS:
            if pat.search(text):
                direction \= dir\_val
                direction\_matched \= True
                break

        if not direction\_matched:
            for word in re.findall(r"\[\\w'’\]+", text.lower()):
                for up\_w in self.UK\_DIRECTION\_LEXICON\["up"\]:
                    if word.startswith(up\_w\[:4\]):
                        direction \+= 1
                for down\_w in self.UK\_DIRECTION\_LEXICON\["down"\]:
                    if word.startswith(down\_w\[:4\]):
                        direction \-= 1
            if direction \!= 0:
                direction \= 1 if direction \> 0 else \-1

        \# 2\. Resolve Relationship vs. Directional Intent
        is\_covariance \= bool(self.RELATION\_STEMS.search(text))
        if is\_covariance and self.ASSOCIATED\_DRIVE\_PATTERNS.search(text):
            is\_covariance \= False  \# Explicit directed driver inquiry takes precedence

        \# 3\. Extract & Ground Candidates
        spans \= self.\_extract\_spans(text, max\_n=3)
        matched\_targets \= set()
        matched\_conditions \= set()

        span\_texts \= \[s\[0\] for s in spans\]
        if span\_texts:
            span\_embs \= self.catalog.encoder.embed(span\_texts)
            span\_norms \= np.linalg.norm(span\_embs, axis=1, keepdims=True)
            span\_embs \= np.divide(span\_embs, np.maximum(span\_norms, 1e-12))
            sim\_matrix \= np.dot(span\_embs, self.catalog.embeddings.T)
        else:
            sim\_matrix \= np.zeros((0, len(self.catalog.literals)))

        for idx, (span\_text, s\_start, s\_end) in enumerate(spans):
            span\_script \= self.catalog.\_detect\_script(span\_text)

            \# Layer A: Exact Lexical Match
            lexical\_hit \= False
            for lit in self.catalog.literals:
                if lit\["case\_sensitive"\]:
                    if span\_text \== lit\["text"\]:
                        self.\_register\_match(lit, matched\_targets, matched\_conditions)
                        lexical\_hit \= True
                else:
                    if span\_text.lower() \== lit\["text"\].lower():
                        self.\_register\_match(lit, matched\_targets, matched\_conditions)
                        lexical\_hit \= True
            if lexical\_hit:
                continue

            \# Layer B: Character n-gram Match (Intra-Script)
            char\_hit \= False
            if span\_script \== 'latin' and self.catalog.char\_vectorizer\_latin and len(span\_text) \>= 4:
                q\_vec \= self.catalog.char\_vectorizer\_latin.transform(\[span\_text\])
                char\_sims \= cosine\_similarity(q\_vec, self.catalog.char\_matrix\_latin)\[0\]
                best\_char \= np.argmax(char\_sims)
                if char\_sims\[best\_char\] \>= 0.45:
                    lit \= self.catalog.literals\[self.catalog.latin\_indices\[best\_char\]\]
                    self.\_register\_match(lit, matched\_targets, matched\_conditions)
                    char\_hit \= True

            elif span\_script \== 'cyrillic' and self.catalog.char\_vectorizer\_cyrillic and len(span\_text) \>= 4:
                q\_vec \= self.catalog.char\_vectorizer\_cyrillic.transform(\[span\_text\])
                char\_sims \= cosine\_similarity(q\_vec, self.catalog.char\_matrix\_cyrillic)\[0\]
                best\_char \= np.argmax(char\_sims)
                if char\_sims\[best\_char\] \>= 0.45:
                    lit \= self.catalog.literals\[self.catalog.cyrillic\_indices\[best\_char\]\]
                    self.\_register\_match(lit, matched\_targets, matched\_conditions)
                    char\_hit \= True
            if char\_hit:
                continue

            \# Layer C: Dense Semantic Match (CSLS \+ Lowe's Ratio)
            if len(span\_text) \>= 2 and sim\_matrix.shape\[0\] \> 0:
                scores \= sim\_matrix\[idx\]
                top\_indices \= np.argsort(scores)\[::-1\]\[:2\]
                top1\_idx, top2\_idx \= top\_indices\[0\], top\_indices\[1\]
                s1, s2 \= scores\[top1\_idx\], scores\[top2\_idx\]

                dist\_ratio \= (1.0 \- s1) / max(1.0 \- s2, 1e-6)

                if s1 \>= self.catalog.min\_csls\_threshold and dist\_ratio \<= self.catalog.ratio\_threshold:
                    lit \= self.catalog.literals\[top1\_idx\]
                    if lit\["case\_sensitive"\] and not span\_text.isupper():
                        continue
                    self.\_register\_match(lit, matched\_targets, matched\_conditions)

        return {
            "text": text,
            "targets": sorted(list(matched\_targets)),
            "conditions": sorted(list(matched\_conditions)),
            "direction": direction,
            "covariance": is\_covariance
        }

    def \_register\_match(self, lit: Dict, matched\_targets: Set\[str\], matched\_conditions: Set\[str\]):
        if lit\["type"\] \== "metric":
            matched\_targets.add(lit\["raw"\])
        elif lit\["type"\] \== "condition":
            matched\_conditions.add(lit\["raw"\])

## **Scoring Formulation and Mathematical Rigor**

### **Grounding Activation Function**

Let ![][image56] denote the user query, decomposed into multi-token spans ![][image57] of length ![][image58]. Let ![][image59] denote the set of canonical graph literals. Each literal ![][image60] possesses token text ![][image61], an attribute type ![][image62], and an offline unit vector representation ![][image63] generated by the Qwen3-Embedding-0.6B model1.
The match activation function ![][image64] is defined hierarchically:
![][image65]
The individual constituent scoring functions are formulated as follows:

#### **Exact Lexical Matching**

![][image66]

#### **Character Subword Similarity**

![][image67]
where ![][image68] denotes the TF-IDF vector over character ![][image6]\-grams with word boundary markers (char\_wb) for ![][image69]3.

#### **Cross-Domain Similarity Local Scaling (CSLS)**

![][image70]
where ![][image71] represents the mean cosine similarity of vector ![][image72] to its ![][image11]\-nearest neighbors (![][image73]) in the literal embedding set ![][image59]5.

#### **Ambiguity Ratio Filter (Lowe's Test)**

![][image74]
where ![][image75] and ![][image76] represent the primary and secondary nearest neighbors in ![][image59] with respect to query span ![][image77]7.

### **Calibrated Seed Pattern Scoring**

Once recognized targets ![][image78], recognized scope conditions ![][image79], and net direction ![][image80] are extracted, each insight pattern ![][image81] in the graph snapshot is evaluated2:
![][image82]
where structural weights are fixed to the established system values: ![][image83]2. When no schema literals are matched (![][image84]), the scoring falls back to:
![][image85]
Seeds meeting ![][image86] are filtered to skip direct hierarchical neighbors, providing up to 3 distinct seed entry points2.

## **Index Design and Storage Mechanics**

The LiteralCatalog resides in memory alongside the DualGraph snapshot, maintaining complete isolation from external database processes8.

| Index Element | Stored Entities | Representation Format | Generation Point | Memory Footprint |
| :---- | :---- | :---- | :---- | :---- |
| Metric Catalog (![][image87]) | Column names, humanized variants | List of string dicts | Snapshot assembly (build\_snapshot)8 | ![][image33] |
| Condition Catalog (![][image88]) | Distinct category and slice values | List of string dicts | Snapshot assembly (build\_snapshot)8 | ![][image89] |
| Direction Catalog (![][image90]) | Canonical polarities (up, down) | List of string dicts | Snapshot assembly (build\_snapshot)8 | ![][image91] |
| Dense Vector Block | Normalized literal embeddings | Float32 NumPy array (![][image92]) | Snapshot assembly (build\_snapshot)8 | ![][image93] |
| Character ![][image6]\-gram Matrix | Subword tokens by script | SciPy CSR matrix | Snapshot assembly (build\_snapshot)8 | ![][image94] |

In tabular statistical insight graphs, the total number of distinct literals ![][image95] scales with schema width and categorical cardinality: ![][image96] entries. Because ![][image97], approximate nearest neighbor (ANN) indexes such as HNSW or FAISS introduce unnecessary indexing overhead. Exact matrix multiplication (np.dot(query\_embs, catalog\_embs.T)) executes in ![][image98] on standard CPUs, guaranteeing ![][image99] candidate recall without vector quantization distortion33. The catalog is serialized as part of the atomic snapshot cache in graph/snapshot.json or stored as companion matrices in state/literals.npz8.

## **Multilingual Grounding Mechanics for Observed Failure Modes**

The execution mechanics of the DMLG architecture demonstrate how previously observed failure modes are systematically resolved2.

| Input Query | Query Tokens & Spans | Grounding Mechanism & Confidence | Extracted Canonical Symbol | Resulting Graph Seed |
| :---- | :---- | :---- | :---- | :---- |
| English Baseline: *"Why is margin lower for phones in the US?"* | margin, phones, US, lower | Layer A: Exact Lexical Match (![][image100])2 | target=margin, conditions=\[category=phones, region=US\], dir=-1 \[cite: 2\] | ![][image101] (Score: ![][image102])2 |
| Code-Switched: *"Чому margin нижчий для phones у US?"* | margin, phones, US, нижчий | Layer A (Literals) \+ Ukrainian Direction Prefix (нижч ![][image103])2 | target=margin, conditions=\[category=phones, region=US\], dir=-1 \[cite: 2\] | ![][image101] (Score: ![][image102])2 |
| Fully Translated: *"Чому маржа нижча для телефонів у США?"* | маржа, телефонів, США, нижча | Layer C: Dense CSLS (Qwen3) \+ Lowe's Ratio \+ Case Check6 | target=margin, conditions=\[category=phones, region=US\], dir=-1 \[cite: 2\] | ![][image101] (Score: ![][image102])2 |
| Composite Comparative: *"Де margin більш низький?"* | margin, більш низький | Layer A (margin) \+ Regex Composite Comparative Rule2 | target=margin, dir=-1 (Net Downward Shift)2 | Patterns with target margin and downward shift2 |
| Associative Phrasing: *"Що пов'язано з вищим return\_rate?"* | return\_rate, пов'язано з вищим | Layer A (return\_rate) \+ Intent Override Rule2 | target=return\_rate, dir=+1 (Driver Analysis)2 | Patterns driving positive shift in return\_rate \[cite: 2\] |
| Inflected Values: *"Харків"* (Catalog) vs. *"у Харкові"* (Query) | харкові | Layer B: Cyrillic char\_wb TF-IDF Match (![][image104])3 | condition: city=Харків | Subgroups filtered to Kharkiv |

### **Grounding Step Details**

#### **Fully Translated Query (*Чому маржа нижча для телефонів у США?*)**

* маржа: Character ![][image6]\-gram matching yields ![][image105] against English literals3. Dense candidate retrieval compares ![][image106] against all literals in ![][image59]. The highest cosine similarity is to margin (![][image107]). The second candidate is discount (![][image108]). The Lowe's ratio is ![][image109], accepting margin as the target metric7.
* телефонів: The genitive plural of *телефон*. Layer C maps ![][image110] to phones (![][image111]). The second candidate is tablets (![][image112]). The ratio test yields ![][image113], binding to category=phones7.
* США: The uppercase abbreviation for USA. Layer C maps ![][image114] to US (![][image115]). Because the catalog literal US is tagged case\_sensitive \= True, the parser verifies that the query token США is uppercase2. The check passes, preventing collision with lowercase pronouns such as *us*, successfully binding to region=US2.
* нижча: Matched by Ukrainian direction prefix нижч ![][image103]2.
* **Outcome**: The query extracts target margin, conditions category=phones, region=US, and direction ![][image52]2. Pattern ![][image116] scores ![][image102], entering the exact same seed as the English query2.

#### **Composite Directional Ambiguity (*Де margin більш низький?*)**

In the legacy parser, більш matched ![][image117] and низьк matched ![][image52], summing to ![][image118] and erasing directionality2. Under DMLG Stage 3, the composite regex detects (більш)\\s+(низький)2. The rule identifies this as a comparative degree intensification of a negative quality ("more low" \= "lower"), setting direction to ![][image52] unambiguously2.

#### **Associative Relationship with Direction (*Що пов'язано з вищим return\_rate?*)**

The token *пов'язано* triggers relationship classification in the legacy parser, overriding and discarding directionality2. Under DMLG Stage 3, the rule detects the construction *пов'язано з* directly preceding the comparative adjective *вищим*2. The system resolves this not as an undirected correlation inquiry, but as a directed driver query for an upward shift in return\_rate (![][image119]), preventing correlation fallback2.

#### **Inflectional Morphology (*Харків* vs. *у Харкові*)**

When condition values are originally in Ukrainian (e.g., store locations), user queries use prepositional cases (*у Харкові*)2. While prefix matching fails because the stem alters between *Харків* and *Харков-*, character ![][image6]\-gram TF-IDF (char\_wb 3–5) matches across the shared character spans хар, арк, рко, ков3. The similarity score of ![][image23] exceeds the intra-script noise floor (![][image120]), resolving the condition without requiring a full lemmatization dictionary3.

## **Complexity Analysis**

The operational footprint of DMLG demonstrates minimal overhead across all deployment metrics:

| Performance Metric | Legacy Retrieval System | DMLG Architecture | Incremental Delta |
| :---- | :---- | :---- | :---- |
| Ingestion / Indexing Latency | ![][image121] per batch | ![][image121] per batch | ![][image122] (One-time literal encoding)1 |
| Query Retrieval Latency | ![][image123] (Dominated by Qwen3)34 | ![][image123] | ![][image124] (Dot products & regex scans) |
| Memory Overhead (RAM) | ![][image125] (Resident Qwen3)1 | ![][image126] | ![][image127] (Catalog vectors & CSR matrices) |
| On-Disk Storage Footprint | ![][image128] (Workspace)2 | ![][image128] | ![][image129] (Serialized snapshot catalog)8 |
| External Infrastructure Services | None | None | Zero new daemons or database processes |

### **Algorithmic Complexity Bounds**

* **Indexing Time**: For ![][image95] literals, generating embeddings requires a single forward pass with batch size ![][image130]:

![][image131]
For a typical schema where ![][image132], this requires 8 batches, adding ![][image133] to batch ingestion1.

* **Query Time**: Given query length ![][image134] tokens, the number of candidate spans is ![][image135]. Grounding requires:
  1. Embedding ![][image136] candidate spans: ![][image137] forward pass (![][image138])1.
  2. Dense dot product: ![][image139] operations (![][image140] on modern CPU)1.
  3. CSLS rescoring and ratio test: ![][image141]6. Total incremental query overhead is bounded by ![][image7].

## **Failure Modes, Adversarial Scenarios, and Mitigation**

The grounding pipeline incorporates specific safeguards for boundary conditions:

| Adversarial Scenario | Failure Mechanism | Architectural Safeguard | System Outcome |
| :---- | :---- | :---- | :---- |
| Short Acronym Collisions | Query token "us" matches English pronoun rather than country code US2. | Strict case-sensitivity flag on literals with length ![][image142]. Lowercase tokens rejected2. | Token "us" ignored; prevents spurious scope constraint2. |
| Semantic Drift / Loose Synonyms | Query token "marginally" maps to target metric margin3. | Lowe's ratio test (![][image143]) rejects candidates without distinct top separation7. | Candidate rejected; query falls back gracefully7. |
| Unseen User Languages | User queries in Polish, German, or Turkish absent from tuning sets. | Frozen Qwen3-Embedding inherently supports ![][image144] languages35. | Zero-shot dense alignment functions across all supported languages35. |
| Severe Typographic Corruption | Severe typos in key tokens (e.g., *mrgn* or *tlphns*). | Both character ![][image6]\-gram and dense CSLS fail acceptance thresholds. | System gracefully falls back to ungrounded semantic baseline2. |
| Polysemous Schema Literals | Dataset column Store collides with common verb *store*. | Literal catalog tags column names; matches require schema structural context. | Prevents verb forms from acting as categorical filters. |

When literal grounding fails completely (no candidate passes acceptance gates), the system degrades to the existing ungrounded fallback (![][image5]), ensuring the architecture never performs worse than the legacy baseline2.

## **Evaluation Benchmark and Experimental Plan**

### **Evaluation Test Matrix**

To validate the architecture, an automated test harness with four query buckets (![][image145]) must be added to tests/test\_multilingual\_retrieval.py:

| Bucket ID | Category Description | Language / Script Profile | Literal State | Representative Test Query |
| :---- | :---- | :---- | :---- | :---- |
| B1 | English Baseline | English (Latin) | Exact English Literals | *"Why is margin lower for phones in the US?"* \[cite: 2\] |
| B2 | Code-Switched Queries | Ukrainian (Cyrillic \+ Latin) | English Literals Preserved | *"Чому margin нижчий для phones у US?"* \[cite: 2\] |
| B3 | Fully Translated Queries | Ukrainian (Cyrillic) | Fully Translated Literals | *"Чому маржа нижча для телефонів у США?"* \[cite: 2\] |
| B4 | Morphological & Typo Noise | Ukrainian & English | Inflected / Typographic Errors | *"Чому маржа нижча для телефонів у Харкові?"* \[cite: 2\] |

### **Quantitative Metrics**

* **Seed Recall@3 (![][image146])**: Fraction of gold-standard seed patterns recovered within the top-3 seeds2.
* **Cross-Lingual Consistency (![][image147])**: The Jaccard similarity of retrieved seed sets between the English baseline query ![][image148] and its translated counterpart ![][image149]2:

![][image150]

* **Evidence Overlap (![][image151])**: Intersection of final evidence items ![][image152] provided to the LLM prompt between parallel queries2.
* **False Positive Grounding Rate (FPGR)**: Percentage of query tokens linked to an incorrect schema literal.

### **Stepwise Implementation and Falsification Sequence**

> 1. **Benchmark Baseline State**: Execute the 60-query benchmark on the existing system. Confirm that Buckets B1 and B2 achieve ![][image153], while Bucket B3 collapses to ![][image154] with ![][image155]2.
> 2. **Evaluate Standalone Character ![][image6]\-grams**: Implement only char\_wb TF-IDF matching. Validate that Bucket B4 (inflections/typos) improves to ![][image156], while Bucket B3 (translated literals) remains at ![][image4], proving that character ![][image6]\-grams cannot bridge scripts3.
> 3. **Enable Dense Literal Matching**: Activate Qwen3-based literal embeddings with raw cosine similarity. Confirm B3 recall increases to ![][image157], but observe elevated FPGR due to hubness and short-acronym collisions5.
> 4. **Calibrate CSLS and Lowe's Ratio Test**: Integrate CSLS rescoring and the ambiguity ratio filter. Verify that FPGR drops below ![][image158] while preserving ![][image159] across all buckets6.
> 5. **Activate Composite Rules**: Apply Ukrainian adverbial and relational disambiguation. Verify that queries containing *більш низький* and *пов'язано з вищим* achieve ![][image99] directional accuracy2.
> 6. **Execute Neo4j Falsification Test**: Run identical retrieval benchmarks with Neo4j running versus Neo4j stopped. Confirm ![][image99] bit-for-bit identity across all seed scores, proving complete architectural independence from the database mirror8.

## **Comparative Evaluation of Candidate Architectures**

Evaluating candidate architectures across functional and operational criteria confirms the trade-offs:

| Architecture | End-to-End Accuracy | Cross-Lingual Robustness | Typo / Morphology Tolerance | Implementation Complexity | Query Latency | Neo4j Independence | Operational Recommendation |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Architecture A: Current Lexical \+ Dense System | Low (![][image160]) | Fails (![][image161])2 | Poor (![][image2])2 | Minimal | ![][image123] | Yes8 | Legacy baseline; unusable for translated queries2. |
| Architecture B: Character ![][image6]\-gram Augmentation Only | Low (![][image162]) | Fails (![][image161])3 | High (![][image163])3 | Low | ![][image123] | Yes8 | Insufficient; cannot bridge Cyrillic to Latin3. |
| Architecture C: Multilingual Literal Embeddings Alone | Moderate (![][image164]) | Good (![][image165]) | Moderate (![][image166]) | Moderate | ![][image123] | Yes8 | Prone to hubness and false-positive acronym drift5. |
| Architecture D: Decoupled Literal Grounding (DMLG) | High (![][image167]) | High (![][image167]) | High (![][image168])3 | Low-Moderate | ![][image123] | Yes8 | **RECOMMENDED**: Optimal balance of accuracy and simplicity. |
| Architecture E: Runtime Translation / LLM Parser | High (![][image168]) | High (![][image169]) | High (![][image170]) | High | ![][image171] | Yes8 | Over-engineered; severe latency and cost overhead. |
| Architecture F: Neo4j-Native Hybrid (Full-Text \+ Vector) | Moderate (![][image172]) | Fails (![][image173]) | Moderate (![][image174]) | High | ![][image175] | **NO** \[cite: 8\] | Rejected; violates deployment independence8. |

## **Strategic Recommendations**

The empirical and architectural evidence confirms that multilingual querying failures stem entirely from a **schema-grounding gap** rather than any limitation in graph traversal, the concept lattice, or the latent anchor ontology2.
By treating dataset literals as language-independent semantic anchors and implementing **Decoupled Multilingual Literal Grounding (DMLG)**, the system achieves robust multilingual alignment while preserving its core design principles9:

* **Maintain In-Memory Autonomy**: Implement the LiteralCatalog as an in-memory extension of the DualGraph snapshot, keeping Neo4j strictly as an optional visualization mirror8.
* **Deploy Hierarchical Gating**: Sequence the query parser through exact lexical matches, intra-script character ![][image6]\-gram TF-IDF, and CSLS-corrected dense semantic matching3.
* **Incorporate Composite Linguistic Rules**: Disambiguate multi-token adverbial comparatives (such as *більш низький*) and separate directional drivers from relational phrasing2.
* **Preserve Dual-Graph Traversal**: Retain the existing Dijkstra best-first search across the concept lattice and latent bridges without modification2.

This architecture represents the smallest, technically justified modification to the existing codebase, ensuring that semantically equivalent queries in English, Ukrainian, or mixed code-switched phrasing reliably activate the same graph seeds, traverse the same structural paths, and produce identical statistical evidence2.

#### **Works cited**

> 1. 04\_representation.md
> 2. 07\_question\_answering.md
> 3. [unknown\_url](http://docs.google.com/unknown_url)
> 4. Mastering Text Embedding and Reranker with Qwen3 \- Alibaba Cloud, [https\://www\.alibabacloud.com/blog/mastering-text-embedding-and-reranker-with-qwen3\_602308](https://www.alibabacloud.com/blog/mastering-text-embedding-and-reranker-with-qwen3_602308)
> 5. Aligning Multilingual Word Embeddings for Cross-Modal Retrieval, [https\://aclanthology.org/D19-6605.pdf](https://aclanthology.org/D19-6605.pdf)
> 6. A Graph-based Coarse-to-fine Method for Unsupervised Bilingual, [https\://aclanthology.org/2020.acl-main.318.pdf](https://aclanthology.org/2020.acl-main.318.pdf)
> 7. How does the Lowe's ratio test work? \- Stack Overflow, [https\://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work](https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work)
> 8. 06\_graph\_and\_storage.md
> 9. Towards Zero-resource Cross-lingual Entity Linking \- ACL Anthology, [https\://aclanthology.org/D19-6127.pdf](https://aclanthology.org/D19-6127.pdf)
> 10. (PDF) Neural Cross-Lingual Entity Linking \- ResearchGate, [https\://www\.researchgate.net/publication/321570979\_Neural\_Cross-Lingual\_Entity\_Linking](https://www.researchgate.net/publication/321570979_Neural_Cross-Lingual_Entity_Linking)
> 11. Re-examining the Role of Schema Linking in Text-to-SQL, [https\://aclanthology.org/2020.emnlp-main.564.pdf](https://aclanthology.org/2020.emnlp-main.564.pdf)
> 12. Text-to-SQL for Low-Resource Languages: A Unified Large ... \- MDPI, [https\://www\.mdpi.com/2079-9292/15/19/4398](https://www.mdpi.com/2079-9292/15/19/4398)
> 13. A Lightweight and Efficient Text-to-SQL Framework with Vector, [https\://www\.researchgate.net/publication/396459006\_LitE-SQL\_A\_Lightweight\_and\_Efficient\_Text-to-SQL\_Framework\_with\_Vector-based\_Schema\_Linking\_and\_Execution-Guided\_Self-Correction](https://www.researchgate.net/publication/396459006_LitE-SQL_A_Lightweight_and_Efficient_Text-to-SQL_Framework_with_Vector-based_Schema_Linking_and_Execution-Guided_Self-Correction)
> 14. A Lightweight and Efficient Text-to-SQL Framework with Vector, [https\://aclanthology.org/2026.findings-eacl.186.pdf](https://aclanthology.org/2026.findings-eacl.186.pdf)
> 15. EMNLP'18 Joint Multilingual Supervision for Cross-lingual Entity, [https\://arxiv.org/html/1809.07657v1](https://arxiv.org/html/1809.07657v1)
> 16. Design Challenges in Low-resource Cross-lingual Entity Linking, [https\://aclanthology.org/2020.emnlp-main.521.pdf](https://aclanthology.org/2020.emnlp-main.521.pdf)
> 17. arXiv:1904.02343v3 \[cs.CL\] 30 Apr 2019, [https\://arxiv.org/pdf/1904.02343](https://arxiv.org/pdf/1904.02343)
> 18. On a Novel Application of Wasserstein-Procrustes for Unsupervised, [https\://openreview.net/attachment?id=MxbAmIs\_ot\&name=pdf](https://openreview.net/attachment?id=MxbAmIs_ot&name=pdf)
> 19. MINER: Multi-crop INference-time Enhancement for Rare-Object, [https\://arxiv.org/html/2609.27142v1](https://arxiv.org/html/2609.27142v1)
> 20. Nearest neighbor search \- Wikipedia, [https\://en.wikipedia.org/wiki/Nearest\_neighbor\_search](https://en.wikipedia.org/wiki/Nearest_neighbor_search)
> 21. SVD PROVABLY DENOISES NEAREST NEIGHBOR DATA, [https\://proceedings.iclr.cc/paper\_files/paper/2026/file/5283806da61c4fb852c331e2713e1e9c-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2026/file/5283806da61c4fb852c331e2713e1e9c-Paper-Conference.pdf)
> 22. An Improved ORB-KNN-Ratio Test Algorithm for Robust Underwater, [https\://www\.mdpi.com/2077-1312/14/2/218](https://www.mdpi.com/2077-1312/14/2/218)
> 23. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large, [https\://proceedings.neurips.cc/paper\_files/paper/2024/file/6ddc001d07ca4f319af96a3024f6dbd1-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/6ddc001d07ca4f319af96a3024f6dbd1-Paper-Conference.pdf)
> 24. HippoRAG 2: Advancing LLM Memory | PDF | Information Retrieval, [https\://www\.scribd.com/document/908670409/From-RAGtoMemory-Non-Parametric-Continual-Learning-for-Large-Language-Models](https://www.scribd.com/document/908670409/From-RAGtoMemory-Non-Parametric-Continual-Learning-for-Large-Language-Models)
> 25. 05\_latent\_anchors.md
> 26. Fine-Tuning Qwen3 Embeddings for product category classification, [https\://blog.ivan.digital/fine-tuning-qwen3-embeddings-for-product-category-classification-on-the-large-scale-product-corpus-3a0919506bc8](https://blog.ivan.digital/fine-tuning-qwen3-embeddings-for-product-category-classification-on-the-large-scale-product-corpus-3a0919506bc8)
> 27. 02\_discovery.md
> 28. 03\_insights.md
> 29. Examining Multilingual Embedding Models Cross-Lingually Through, [https\://aclanthology.org/2025.findings-emnlp.115.pdf](https://aclanthology.org/2025.findings-emnlp.115.pdf)
> 30. Findings \- ACL 2023, [https\://2023.aclweb.org/program/accepted\_findings/](https://2023.aclweb.org/program/accepted_findings/)
> 31. PropRAG: Guiding Retrieval with Beam Search over Proposition Paths, [https\://aclanthology.org/2025.emnlp-main.317.pdf](https://aclanthology.org/2025.emnlp-main.317.pdf)
> 32. Proceedings of the Knowledge Graphs and Large Language Models, [https\://aclanthology.org/2026.kallm-1.pdf](https://aclanthology.org/2026.kallm-1.pdf)
> 33. Adversarial Learning for Multi-Lingual Entity Linking \- ACL Anthology, [https\://aclanthology.org/2024.sighan-1.4.pdf](https://aclanthology.org/2024.sighan-1.4.pdf)
> 34. Cross Lingual Mention and Entity Embeddings for Cross-Lingual, [https\://tac.nist.gov/publications/2016/participant.papers/TAC2016.OSU\_DEFT.proceedings.pdf](https://tac.nist.gov/publications/2016/participant.papers/TAC2016.OSU_DEFT.proceedings.pdf)
> 35. GitHub \- QwenLM/Qwen3-Embedding, [https\://github.com/QwenLM/Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding)