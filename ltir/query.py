"""Query interpretation and seed resolution (docs/07_question_answering.md §7.1–7.2).

Deterministic parse against the graph vocabulary (metrics, dimension values)
plus a projection of the question into the insight space with the same
tripartite composition as patterns. No LLM is involved in retrieval.

Literal grounding (§7.1.1): what the lexical pass leaves of a question in any language is
matched span by span onto the graph's literal catalog — same-script character n-grams, then
the multilingual embedding with a local-margin and ratio gate — so *маржа*, *телефонів* and
*США* become ``margin``, ``category=phones`` and ``region=US`` before the structural seed
scoring runs. Literals are never translated: the catalog holds them as the data does.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ltir.canonical import covariance_label, humanize
from ltir.config import Config
from ltir.encoder import InsightEncoder, l2_normalize
from ltir.graph import DualGraph
from ltir.models import Insight

NEG_WORDS = frozenset(
    "lower low lowest decrease decreases decreased decreasing decline declines declining drop drops dropped down fall falls falling less reduced reduce reduction worse worst negative erosion eroded eroding compressed compression shrink shrinking below weak weaker poor smaller shorter fewer cheaper loss losses".split()
)
POS_WORDS = frozenset(
    "higher high highest increase increases increased increasing rise rises rising up more grow grows growth uplift better best positive above elevated spike surge larger longer bigger delay delays delayed slow slower stronger inflated".split()
)
# Ukrainian questions: inflected forms are matched by stem (prefix). Data literals (column names,
# category values) are matched as typed, in whatever script the data holds them (§7.1.1).
NEG_STEMS = ("нижч", "низьк", "менш", "пада", "спад", "знижен", "знижу", "зменш", "слабш", "гірш", "скороч", "втрат", "дешевш", "коротш")
POS_STEMS = ("вищ", "висок", "більш", "зрост", "збільш", "підвищ", "затрим", "повільн", "сильніш", "кращ", "довш", "дорожч")
WEAKENING_WORDS = ("break", "weak", "decoupl", "lose", "loss", "disappear", "руйн", "слаб", "ослаб", "послаб", "розпад", "зник", "розрив", "втрач")
GENERIC_METRIC_WORDS = {"median", "mean", "average", "avg", "total", "number", "num", "count", "percent", "pct", "rate", "value", "score", "index"}
COVARIANCE_WORDS = tuple(
    "correl relationship relation coupl decoupl dependen covari linked associat кореляц зв'яз пов'яз взаємозв залежн асоці".split()
)
_TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)*")  # letters and digits in any script; underscores split (return_rate -> return, rate)

# --- literal grounding (§7.1.1) ---------------------------------------------------------------
# A comparative adverb before a direction adjective is one signed vote: "більш низький" = lower.
COMPARATIVE_ADVERBS = {"більш": 1, "більше": 1, "менш": -1, "менше": -1, "more": 1, "less": -1}
# "associated with higher X" asks what drives X up: a directed question, not a relationship one.
DRIVER_VERBS = ("пов'яз", "associat", "linked", "related")
DRIVER_PREPOSITIONS = {"з", "із", "зі", "with", "to"}
STOPWORDS = frozenset(
    "why what where which when how does do is are the a an for in on at of to about tell me show and or with by from than there their this that it its "
    "чому що де який яка які коли як для у в на і та з із зі при про від до це чи має є також там їх цей ця ці ніж між розкажи скажи покажи поясни".split()
)
CHAR_MIN = 0.30  # char_wb (3–5) TF-IDF cosine floor (measured: inflections >= 0.33, typos >= 0.36, distractors <= 0.23)
CHAR_MAX_LEN_DIFF = 3  # a same-script surface variant stays about as long as the literal (marginally -> margin is 0.89 but 4 letters longer)
MARGIN_MIN, MARGIN_K, LOWE_MAX = 0.15, 5, 0.85  # dense gates: local margin over the catalog, Lowe's ratio
MAX_SPAN, ACRONYM_MAX_LEN = 3, 4


def detect_script(text: str) -> str:
    for ch in text:
        name = unicodedata.name(ch, "")
        if "CYRILLIC" in name:
            return "cyrillic"
        if "LATIN" in name:
            return "latin"
    return "other"


@dataclass
class LiteralCatalog:
    """The graph's literals as the data holds them, indexed for grounding (§7.1.1): metric names (raw
    and humanised) and condition values (with the attributes they occur under); unit embeddings, and
    one char_wb TF-IDF index per script. Built by ``build_catalog`` (``Engine.catalog`` keeps the
    vectors in ``graph/literals.npz``, so a known literal is never embedded twice)."""

    texts: list[str]
    types: list[str]  # metric | condition
    symbols: list[Any]  # metric name | list of attributes
    case_sensitive: list[bool]
    vectors: np.ndarray
    chars: dict[str, Any] = field(default_factory=dict)  # script -> (TfidfVectorizer, matrix, indices)
    embedder: Any = field(default=None, repr=False, compare=False)  # the model that embedded the catalog (attached by the engine)

    def __post_init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        # the embedding space is anisotropic (every literal is 0.6–0.85 cosine to every other): centring by the
        # catalog mean and re-normalising makes cosines discriminative (measured in docs/07 §7.1.1)
        self.centre = self.vectors.mean(axis=0)
        self.centred = l2_normalize(self.vectors - self.centre)
        self.scripts = [detect_script(t) for t in self.texts]
        self.words = np.array([len(t.split()) for t in self.texts])
        by_script: dict[str, list[int]] = {}
        for i, script in enumerate(self.scripts):
            by_script.setdefault(script, []).append(i)
        for script, idx in by_script.items():
            if script != "other":
                vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
                self.chars[script] = (vec, vec.fit_transform([self.texts[i] for i in idx]), idx)

    def symbol_key(self, j: int) -> str:
        return f"{self.types[j]}:{json.dumps(self.symbols[j], ensure_ascii=False)}"


def _values(graph: DualGraph) -> dict[str, list[str]]:
    """Every condition value of the graph (first-seen order) -> the attributes it occurs under."""
    values: dict[str, set[str]] = {}
    for ins in graph.insights.values():
        for c in ins.conditions:
            values.setdefault(c.value, set()).add(c.attribute)
    return {v: sorted(a) for v, a in values.items()}


def build_catalog(graph: DualGraph, embedder: Any, known: dict[str, np.ndarray] | None = None) -> LiteralCatalog | None:
    """The graph's literals, embedded with the document-side embedder (as component labels are); ``known`` maps a
    literal to its unit vector from an earlier build, so only new literals are embedded. ``None`` for an empty graph."""
    if not graph.insights:
        return None
    texts, types, symbols, case = [], [], [], []
    for name in sorted({n["props"]["name"] for n in graph.of_kind("Metric")}):
        for text in dict.fromkeys((name, humanize(name))):  # raw and humanised form (once when equal)
            texts.append(text), types.append("metric"), symbols.append(name), case.append(False)
    for value, attrs in sorted(_values(graph).items()):
        texts.append(value), types.append("condition"), symbols.append(attrs), case.append(len(value) <= ACRONYM_MAX_LEN and value.isupper())
    vectors = dict(known or {})
    missing = [t for t in dict.fromkeys(texts) if t not in vectors]
    if missing:
        vectors.update(zip(missing, l2_normalize(embedder.embed(missing))))
    return LiteralCatalog(texts, types, symbols, case, np.stack([vectors[t] for t in texts]).astype(np.float32), embedder=embedder)


def _stem_match(a: str, b: str) -> bool:
    if a == b:
        return True
    k = min(len(a), len(b))
    return k >= 5 and a[:5] == b[:5] and abs(len(a) - len(b)) <= 3


@dataclass
class ParsedQuery:
    text: str
    targets: list[str] = field(default_factory=list)  # metric names
    direction: int = 0
    conditions: list[tuple[str, str]] = field(default_factory=list)  # (attribute, value); attribute "*" = any column holding the value
    covariance: bool = False  # question is about a relationship between metrics
    grounding: list[dict[str, Any]] = field(
        default_factory=list
    )  # spans resolved by the literal catalog (§7.1.1): span, literal, layer, score, symbol

    @property
    def is_lexical(self) -> bool:
        return bool(self.targets or self.conditions)

    def components(self) -> tuple[tuple[str, float], ...]:
        if self.covariance and len(self.targets) >= 2:
            weakening = any(t.startswith(WEAKENING_WORDS) for t in _TOKEN.findall(self.text.lower()))  # a word, not a substring ("close")
            return ((covariance_label(self.targets[:2]), -2.0 if weakening else 2.0),)
        if not self.direction:
            # no direction word: no signed phenomenon components; the encoder then falls back
            # to the question text for the phenomenon block instead of assuming "higher"
            return ()
        return tuple((humanize(t), float(self.direction) * 2.0) for t in self.targets)

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "conditions": [f"{a}={v}" if a != "*" else v for a, v in self.conditions]}


def _is_direction(token: str) -> int:
    if token in POS_WORDS or token.startswith(POS_STEMS):
        return 1
    if token in NEG_WORDS or token.startswith(NEG_STEMS):
        return -1
    return 0


def _direction(tokens: list[str]) -> int:
    """sign(#up − #down); a comparative adverb before a direction adjective casts one vote with the adjective
    (``більш низький`` = lower, ``менш високий`` = lower) instead of two that cancel."""
    votes, i = 0, 0
    while i < len(tokens):
        t = tokens[i]
        if t in COMPARATIVE_ADVERBS and i + 1 < len(tokens) and _is_direction(tokens[i + 1]):
            votes += COMPARATIVE_ADVERBS[t] * _is_direction(tokens[i + 1])
            i += 2
            continue
        votes += _is_direction(t)
        i += 1
    return int(np.sign(votes))


def _directed_driver(tokens: list[str]) -> bool:
    """``пов'язано з вищим X`` / ``associated with higher X``: a directed driver question, not a relationship one."""
    return any(
        tokens[i].startswith(DRIVER_VERBS) and i + 2 < len(tokens) and tokens[i + 1] in DRIVER_PREPOSITIONS and _is_direction(tokens[i + 2])
        for i in range(len(tokens))
    )


def _spans(raw_tokens: list[str], tokens: list[str], claimed: set[int]) -> list[tuple[str, set[int]]]:
    """Candidate spans, longest first, over unclaimed tokens; stop-word-only and one-letter spans are pruned,
    acronyms (short upper-case tokens) never are."""
    out = []
    for n in range(MAX_SPAN, 0, -1):
        for i in range(len(tokens) - n + 1):
            pos = set(range(i, i + n))
            if pos & claimed:
                continue
            words = tokens[i : i + n]
            acronym = n == 1 and len(raw_tokens[i]) <= ACRONYM_MAX_LEN and raw_tokens[i].isupper()
            if not acronym and (words[0] in STOPWORDS or words[-1] in STOPWORDS or (n == 1 and len(words[0]) <= 1)):
                continue  # a span neither starts nor ends with a stop word: "для телефонів у" never competes with "телефонів"
            out.append((" ".join(raw_tokens[i : i + n]), pos))
    return out


def _ground(q: ParsedQuery, catalog: LiteralCatalog, raw_tokens: list[str], tokens: list[str], claimed: set[int], config: Config | None) -> None:
    """Resolve the spans the lexical pass left onto catalog literals: same-script character n-grams first,
    then the multilingual embedding behind the floor / local-margin / ratio / case gates. An accepted
    span claims its tokens (non-maximum suppression), so sub-spans never ground again."""
    floor = config.grounding_min_cosine if config else 0.30
    pending: list[tuple[str, set[int]]] = []
    for span, pos in _spans(raw_tokens, tokens, claimed):
        if pos & claimed:
            continue
        hit = None
        if len(span) >= 4 and (entry := catalog.chars.get(detect_script(span))):
            vec, matrix, idx = entry
            sims = np.where(catalog.words[idx] == len(span.split()), (matrix @ vec.transform([span]).T).toarray().ravel(), -1.0)
            best = int(np.argmax(sims))
            if sims[best] >= CHAR_MIN and abs(len(span) - len(catalog.texts[idx[best]])) <= CHAR_MAX_LEN_DIFF:
                hit = (idx[best], "chars", float(sims[best]))
        if hit is not None:
            _register(q, catalog, hit, span, pos, claimed, tokens)
        else:
            pending.append((span, pos))
    if not pending:
        return
    vectors = l2_normalize(l2_normalize(catalog.embedder.embed([s for s, _ in pending])) - catalog.centre)
    cos = vectors @ catalog.centred.T
    for (span, pos), scores in zip(pending, cos):
        if pos & claimed:
            continue
        # the dense layer bridges scripts (маржа -> margin); same-script variation is layer B's job, so a Latin span
        # never grounds densely onto a Latin literal (marginally -> margin, sales -> retail)
        # ... and a span of n words grounds onto a literal of about n words: "телефонів у США" is not "phones";
        # its single words are tried next ("дні доставки" -> "delivery days" is what multi-word spans are for)
        script = detect_script(span)
        allowed = [s != script and abs(w - len(span.split())) <= 1 for s, w in zip(catalog.scripts, catalog.words)]
        scores = np.where(allowed, scores, -np.inf)
        order = np.argsort(scores)[::-1]
        j, s1 = int(order[0]), float(scores[order[0]])
        if not np.isfinite(s1) or s1 < floor:
            continue
        others = [i for i in order[1:] if np.isfinite(scores[i]) and catalog.symbol_key(i) != catalog.symbol_key(j)]
        s2 = float(scores[others[0]]) if others else -1.0  # Lowe: the runner-up must mean something else (not the raw/humanised twin)
        top = scores[order[: min(MARGIN_K, len(order))]]
        margin = s1 - float(top[np.isfinite(top)].mean())
        if margin < MARGIN_MIN or (1.0 - s1) / max(1.0 - s2, 1e-6) > LOWE_MAX or (catalog.case_sensitive[j] and not span.isupper()):
            continue
        _register(q, catalog, (j, "dense", s1), span, pos, claimed, tokens)


def _bind(q: ParsedQuery, value: str, attrs: list[str], pos: set[int], tokens: list[str]) -> list[str]:
    """Add the condition of ``value`` found at ``pos``. A value under several columns binds the one named within
    two tokens of it, else ``*`` (any column holding the value, never a scope conflict). Returns the columns bound."""
    if len(attrs) > 1:
        near = [tokens[i] for i in range(max(0, min(pos) - 2), min(len(tokens), max(pos) + 3)) if i not in pos]
        named = [a for a in attrs if any(_stem_match(part, t) for part in humanize(a).lower().split() for t in near)]
        attrs = named if len(named) == 1 else ["*"]
    for attr in attrs:
        if (attr, value) not in q.conditions:
            q.conditions.append((attr, value))
    return attrs


def _register(q: ParsedQuery, catalog: LiteralCatalog, hit: tuple[int, str, float], span: str, pos: set[int], claimed: set[int], tokens) -> None:
    j, layer, score = hit
    text, symbol = catalog.texts[j], catalog.symbols[j]
    if catalog.types[j] == "metric":
        if symbol not in q.targets:
            q.targets.append(symbol)
    else:
        symbol = [f"{a}={text}" if a != "*" else text for a in _bind(q, text, symbol, pos, tokens)]
    claimed |= pos
    q.grounding.append({"span": span, "literal": text, "layer": layer, "score": round(score, 3), "symbol": symbol})


def parse_query(text: str, graph: DualGraph, config: Config | None = None, catalog: LiteralCatalog | None = None) -> ParsedQuery:
    """Targets, conditions, direction and relationship intent of a question. Literals typed as the data holds
    them match lexically; with ``catalog`` (``Engine.catalog``) the rest of the question is grounded (§7.1.1)."""
    raw_tokens = _TOKEN.findall(text.replace("’", "'").replace("ʼ", "'"))
    tokens = [t.lower() for t in raw_tokens]
    q = ParsedQuery(text=text)
    claimed: set[int] = set()  # token positions taken by a literal, then by direction and relationship words

    metrics = sorted({n["props"]["name"] for n in graph.of_kind("Metric")})
    parts_of = {m: [p for p in humanize(m).lower().split() if len(p) >= 3] for m in metrics}
    head_count: dict[str, int] = {}
    for parts in parts_of.values():
        if parts:
            head_count[parts[0]] = head_count.get(parts[0], 0) + 1
    matched: list[tuple[float, str]] = []
    for name, parts in parts_of.items():
        if not parts:
            continue
        hit = [any(_stem_match(p, t) for t in tokens) for p in parts]
        specific = sum(h for h, p in zip(hit, parts) if p not in GENERIC_METRIC_WORDS)
        # full name; or two words incl. a specific one ("house values" -> median_house_value);
        # or a distinctive head word ("returns" -> return_rate). "median" alone matches nothing.
        distinctive_head = hit[0] and parts[0] not in GENERIC_METRIC_WORDS and head_count[parts[0]] == 1
        if all(hit) or (sum(hit) >= 2 and specific >= 1) or distinctive_head:
            matched.append((sum(hit) / len(parts), name))
    if matched:  # best-covered metrics win (a full-name match beats partial ones)
        best = max(score for score, _ in matched)
        q.targets = sorted(name for score, name in matched if score == best)
        claimed |= {i for i, t in enumerate(tokens) for name in q.targets for p in parts_of[name] if _stem_match(p, t)}

    for value, attrs in _values(graph).items():
        if len(value) <= ACRONYM_MAX_LEN and value.isupper():  # an acronym matches as typed: "us" is not "US"
            pos = {i for i, t in enumerate(raw_tokens) if t == value}
        else:  # every word of the value, inflected or not
            found = [{i for i, t in enumerate(tokens) if _stem_match(v, t)} for v in value.lower().split()]
            pos = set().union(*found) if all(found) else set()
        if pos:
            claimed |= pos
            _bind(q, value, attrs, pos, tokens)

    free = ["" if i in claimed else t for i, t in enumerate(tokens)]  # a literal's words are not intent words ("delivery delay")
    q.direction = _direction(free)
    q.covariance = any(t.startswith(COVARIANCE_WORDS) for t in free) and not _directed_driver(free)
    if q.covariance:
        q.direction = 0  # "breaks down" / "weakens" describe the relationship, not a metric level
    claimed |= {i for i, t in enumerate(tokens) if _is_direction(t) or t.startswith(COVARIANCE_WORDS) or t in COMPARATIVE_ADVERBS}
    if catalog is not None:
        _ground(q, catalog, raw_tokens, tokens, claimed, config)
        q.targets = sorted(q.targets)
    return q


@dataclass
class SeedMatch:
    pattern_id: str
    score: float
    matched: dict[str, Any]


def score_pattern(ins: Insight, query: ParsedQuery, qvec: np.ndarray, vec: np.ndarray | None, config: Config) -> SeedMatch:
    """Seed score of one pattern: lexical match (target, scope, direction) + semantic cosine + weight."""
    pair = set(ins.covariance.get("pair", []))
    wanted_targets = set(query.targets)
    material = [s.robust_z for t in query.targets for s in ins.shifts if s.metric == t and s.magnitude >= config.min_component_z]
    if ins.target in wanted_targets:
        target = 1.0
    elif material:
        target = 0.7
    elif pair & wanted_targets:
        target = 0.6
    else:
        target = 0.0
    if query.covariance:  # relationship questions: the phenomenon slot scores covariance insights
        direction = 1.0 if ins.phenomenon_type == "covariance" and (not query.targets or pair & wanted_targets) else 0.0
    elif query.direction and material:
        direction = 1.0 if np.sign(material[0]) == query.direction else -0.5
    else:
        direction = 0.0
    wanted = set(query.conditions)
    exact = {(a, v) for a, v in wanted if a != "*"}
    wanted_attrs = {a for a, _ in exact}
    conds = {(c.attribute, c.value) for c in ins.conditions}
    hits = len(conds & exact) + sum(any(v == cv for _, cv in conds) for a, v in wanted if a == "*")  # a wildcard matches any column
    conflicts = len({a for a, v in conds if a in wanted_attrs and (a, v) not in exact})
    scope = (hits / len(wanted) if wanted else 0.0) - 0.5 * conflicts
    semantic = float(vec @ qvec) if vec is not None else 0.0
    if query.is_lexical:
        score = 0.35 * target + 0.25 * scope + 0.15 * direction + 0.15 * semantic + 0.10 * ins.weight
    else:
        score = 0.7 * semantic + 0.3 * ins.weight
    return SeedMatch(
        ins.id,
        float(score),
        {
            "target": target,
            "scope": round(scope, 3),
            "direction": direction,
            "semantic": round(semantic, 3),
            "weight": round(ins.weight, 3),
            "scope_conflicts": conflicts,
        },
    )


def resolve_seeds(
    query: ParsedQuery,
    graph: DualGraph,
    encoder: InsightEncoder,
    pattern_vectors: dict[str, np.ndarray],
    config: Config,
) -> list[SeedMatch]:
    """Score every Pattern against the parsed query and pick diverse top seeds."""
    scope_text = "; ".join(f"{a} = {v}" if a != "*" else v for a, v in query.conditions)
    target_text = "; ".join(humanize(t) for t in query.targets)
    qvec = encoder.encode_query(scope_text, target_text, query.components(), query.text)
    scored = sorted(
        (score_pattern(ins, query, qvec, pattern_vectors.get(pid), config) for pid, ins in graph.insights.items()),
        key=lambda s: -s.score,
    )
    seeds: list[SeedMatch] = []
    floor = max(config.seed_min_score, config.seed_relative_min * scored[0].score) if scored else 1.0
    for cand in scored:  # diverse seeds: skip direct lattice neighbours of chosen seeds
        if len(seeds) >= config.seed_top_k or cand.score < floor:
            break
        near = {o for s in seeds for _, o in graph.incident(s.pattern_id, ["SPECIALIZES", "GENERALIZES"])}
        if cand.pattern_id not in near:
            seeds.append(cand)
    if not seeds and scored:
        seeds = [max(scored, key=lambda s: s.matched["semantic"])]
        seeds[0].matched["fallback"] = "semantic"
    return seeds
