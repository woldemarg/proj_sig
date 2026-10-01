// Mutual k-NN topology is recomputed every batch: drop the previous RELATED_TO set before MERGE.
MATCH (:Attractor)-[r:RELATED_TO]->(:Attractor) DELETE r
