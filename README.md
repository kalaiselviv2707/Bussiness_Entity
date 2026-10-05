# ENTITYRESOLVE AI – Business Entity Resolution

Amazon ML Challenge: decide which noisy Source 2 / Source 3 business records describe the same
real-world business as each Source 1 (reference) entity. Scored with **F0.5**, so precision matters
more than recall; entities without a confident match get an empty result.
No external lookup of any kind (web search, maps, business APIs, geocoding) is used.

## Architecture
```
TSV data -> preprocessing -> blocking (candidates) -> pair features -> ML model -> F0.5 threshold -> outputs -> dashboard
```
* **Data**: `train/` and `test/` folders with `*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`
  (columns `entity_id, business_name, business_address, country`) and `train_ground_truth.tsv`
  (`source1_entity_id, matched_entity_ids`). Always read as TSV. Only the **first 10,000 records** of every
  source are used (`LIMIT = 10000`, change with env var `ER_LIMIT`). Source 2 and 3 are searched together.
* **Preprocessing** (`src/preprocessing.py`): NFKC, accent stripping, lowercase, punctuation/whitespace,
  `&`→`and`, domain names reduced to their name, legal suffixes (inc, llc, pvt, ltd, sarl, …) removed from a
  *core* name, address abbreviation expansion (st→street …), noise words (unit, apt …) dropped, US state
  codes expanded, **rule-based transliteration of Hindi/Tamil/Kannada/other Indic scripts** to Latin.
  Original columns are kept next to the normalised ones. Countries are never hardcoded.
* **Candidate generation** (`src/features.py`): only same-country pairs; union of top-k lists from five
  sparse TF-IDF signals (name char n-grams, rare-token name cosine, space-insensitive name n-grams,
  address n-grams incl. street numbers, combined name+address). Chunked sparse matrix products – no
  all-pairs loops. During training, true matches missed by blocking are injected (training only).
* **Features (32)**: name Jaccard/overlap/shared tokens/char-n-gram/TF-IDF cosine/exact/length/first token/prefix
  (abbreviation) ratio; address Jaccard/overlap/char-n-gram/TF-IDF/numeric overlap/length/missing/exact;
  country match/mismatch; source indicator; combined scores; rank/gap of the pair inside the entity's candidate list.
* **Models**: RandomForest, HistGradientBoosting, LogisticRegression are compared; the best **validation F0.5** wins.
* **Validation**: split grouped by Source 1 entity (no leakage); recall counts every true pair, including those
  blocking missed; threshold searched 0.05–0.95 for maximum F0.5 (middle of the best plateau). Model, threshold and
  metrics are saved (`model.pkl`, `output/model_metrics.json`).
* **Matching**: probabilities for all candidates in one batch; every candidate ≥ threshold is a match
  (zero, one or many). Nothing is forced.

## Installation
```
py -m pip install -r requirements.txt
```
Place the data in `dataset/train/` and `dataset/test/` (or set `ER_DATA_DIR`).

## Run
```
python src/train_model.py        # trains, prints P / R / F0.5 / threshold / TP FP FN, writes model.pkl + metrics
python src/predict.py            # writes output/matching_results.tsv and output/candidate_pairs.tsv
python src/validate_outputs.py   # checks format, IDs, duplicates, matches ⊆ candidates
python src/main.py               # all three steps
python dashboard_server.py       # dashboard at http://127.0.0.1:5000
python build_submission.py       # validates and builds Elite_Warriors_Submission.zip
```

## Outputs
* `output/matching_results.tsv` – `source1_entity_id`, `matched_entity_ids` (comma separated, empty = no match)
* `output/candidate_pairs.tsv` – `source1_entity_id`, `candidate_entity_ids` (exactly the candidates scored by the model)
* `output/model_metrics.json` – validation metrics, model comparison, training/prediction summary (used by the dashboard)

## Dashboard
Flask (`dashboard_server.py`) serves `dashboard/` and the JSON API: `/api/dashboard`, `/api/matches`
(`status`, `q`, `page`, `page_size`), `/api/search?q=`, `/api/candidates`, `/api/model-metrics`, `/api/health`.
Everything is read from the generated output files (cached, reloaded when the files change). Charts use Chart.js from a CDN
(internet needed for the charts only). Missing or malformed files produce a clear error message.

## Project structure
```
src/{config,preprocessing,features,train_model,predict,main,validate_outputs}.py
dashboard/{index.html,style.css,script.js}   dashboard_server.py   build_submission.py
requirements.txt   README.md   Documentation_template.md
```
