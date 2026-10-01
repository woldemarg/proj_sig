// Transversal traversal in Neo4j Browser (mirrors ltir/traversal.py, docs/07_question_answering.md §7.3).
// Seed Pattern -> ACTIVATES -> Attractor -> RELATED_TO{0,1} -> Attractor <- ACTIVATES <- Pattern,
// then one lattice hop (SPECIALIZES / GENERALIZES / CONTRASTS). Best path per target. Replace the seed id.
// Pattern nodes are the Insight record: weight is `weight`. Nested fields (conditions, shifts,
// canonical) are stored as <field>_json.
:param seed => 'P-bc4657a04746';
:param min_act => 0.40;
:param min_rel => 0.40;

MATCH (seed:Pattern {id: $seed})-[up:ACTIVATES]->(a1:Attractor)
WHERE up.weight >= $min_act
OPTIONAL MATCH (a1)-[rel:RELATED_TO]-(a2:Attractor)
WHERE rel.weight >= $min_rel
WITH seed, up, a1, [{a: a1, w: 1.0}] + [x IN collect({a: a2, w: rel.weight}) WHERE x.a IS NOT NULL] AS hops
UNWIND hops AS hop
MATCH (target:Pattern)-[down:ACTIVATES]->(anchor:Attractor)
WHERE anchor = hop.a AND target <> seed AND down.weight >= $min_act
OPTIONAL MATCH (target)-[lat:SPECIALIZES|GENERALIZES|CONTRASTS]->(nbr:Pattern)
WITH seed, a1, anchor, target, up.weight * hop.w * down.weight * target.weight AS score,
     collect(DISTINCT type(lat) + ' ' + nbr.expression) AS lattice
ORDER BY score DESC
WITH target, collect({seed: seed, a1: a1, anchor: anchor, score: score, lattice: lattice})[0] AS best
RETURN best.seed.expression AS seed, best.a1.label AS primary_anchor, best.anchor.label AS reached_anchor,
       target.id AS pattern, target.expression AS expression, target.target AS target_metric,
       round(best.score * 1000) / 1000 AS score,
       best.lattice AS lattice
ORDER BY score DESC
LIMIT 25;
