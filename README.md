# Financial Network Risk Propagation

Analytics and ML for systemic risk in a multi-layer financial network: which institutions could set off a cascade of
failures if they got into trouble, and why.

The network has **4,615 entities** (100 large banks, 4,376 FDIC banks, 136 listed stocks, 3 institutional investors) and
**19,411 exposures** across 15 relationship types (interbank lending, repos, derivatives, credit lines, equity
ownership, market correlation and more). The CSVs in `data/` are in Neo4j's bulk-import format, so the same files load
straight into a Neo4j graph.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `streaming_algorithms/run_streaming_pipeline.py` | reads the edges as a stream and runs a **Bloom filter** (duplicate exposures), **DGIM** (count of $1B+ exposures in a sliding window), **Flajolet-Martin** (distinct counterparties) and **reservoir sampling** |
| 2 | `lsh/entity_similarity.py` | **MinHash LSH** (100 hashes, 20 bands) finds entities with near-identical counterparty sets |
| 3 | `graph_analytics/network_features.py` | PageRank, degree, betweenness, clustering coefficient and total exposure per entity |
| 4 | `ml_models/train_cascade_predictor.py` | builds cascade labels with a contagion simulation, trains a **Random Forest** (200 trees, class-balanced) |
| 5 | `ml_models/model_inference.py` | scores every entity's cascade risk |
| 6 | `explainability/shap_explainer.py` | **SHAP** feature importance and per-entity explanations |
| 7 | `explainability/fairness_audit.py` | compares predicted cascade rates across entity types and tiers (disparate-impact ratio) |
| 8 | `explainability/model_unlearning.py` | removes chosen entities from the training data and retrains (machine unlearning) |

## Results (current data)

- Stream: 19,411 exposures; 5,798 repeat the same source-target pair. The Bloom filter (sized for the stream at a 1%
  target false-positive rate: 186,056 bits, 7 hashes) flagged 5,805, so 7 false positives.
- 4,741 exposures above $1B; about 330 distinct sources and 2,647 distinct targets (Flajolet-Martin estimates).
- Cascade model: F1 0.996, ROC-AUC 0.998 on the held-out split.

**Read the model scores with care:** the cascade labels come from a contagion simulation over the same network
features the model learns from, and only about 3-4% of entities end up labelled "no cascade". The high scores show the model
recovers the simulation; they are not a forecast of real bank failures.

## Run it

Requires Python 3.10+.

```bash
git clone https://github.com/yuktaablitz/financial_network_risk_propogation.git
cd financial_network_risk_propogation
python3 -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

python streaming_algorithms/run_streaming_pipeline.py
python lsh/entity_similarity.py
python graph_analytics/network_features.py
python ml_models/train_cascade_predictor.py
python ml_models/model_inference.py
python explainability/shap_explainer.py
python explainability/fairness_audit.py
python explainability/model_unlearning.py
```

Run them in this order from the repo root; each step writes to `output/` and later steps read from it. The whole run
takes a few minutes on a laptop.

### Load the graph into Neo4j (optional)

```bash
neo4j-admin database import full --nodes=data/nodes.csv --relationships=data/edges.csv neo4j
```

## Repo layout

```
data/                    nodes.csv, edges.csv (Neo4j bulk-import format)
streaming_algorithms/    Bloom filter, DGIM, Flajolet-Martin, reservoir sampling, stream runner
lsh/                     MinHash LSH and entity similarity
graph_analytics/         network features
ml_models/               training and inference
explainability/          SHAP, fairness audit, machine unlearning
output/                  results, features, models, predictions, explanations
```

## Tech stack

Python, pandas, NumPy, scikit-learn, SHAP, matplotlib, mmh3, bitarray, Neo4j import format.
