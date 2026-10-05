# Business Entity Resolution – Documentation

## 1. Team Information
* Team name: Elite Warriors
* Members: _<fill in names / IDs>_

## 2. Problem Understanding
Three independent business sources: Source 1 (deduplicated reference) and Source 2 / Source 3 (noisy records).
For each Source 1 entity the system outputs the Source 2/3 records describing the same business: zero, one or
many. Scoring is F0.5, so false positives are penalised more than misses. No external lookup is allowed.

## 3. Approach
Preprocess → block into candidates → compute pair features → classify with a lightweight model →
apply an F0.5-optimal threshold. Only the first {{LIMIT}} records of every source are processed.

## 4. Data Preprocessing
Unicode (NFKC) and accent normalisation, lowercase, punctuation/whitespace cleanup, legal-suffix removal for a
core name, domain-name reduction, address abbreviation expansion and noise-word removal, US state-code expansion,
rule-based transliteration of Indic scripts (Devanagari, Tamil, Kannada, …) to Latin. Original text is retained.

## 5. Candidate Generation
Same-country restriction plus the union of top-k lists from name character n-grams, rare-token name cosine,
space-insensitive name n-grams, address n-grams and a combined score, using chunked sparse matrix products.
Training run: {{TRAIN_CANDIDATES}} candidate pairs, blocking recall on reachable true pairs {{BLOCKING_RECALL}}
({{TRUE_IN_SCOPE}} true pairs inside the first {{LIMIT}} records; {{TRUE_OUTSIDE}} true pairs point to records
outside that window and cannot be found).

## 6. Feature Engineering
{{FEATURE_COUNT}} features: name similarity (Jaccard, overlap, shared tokens, char n-gram and TF-IDF cosine,
exact, length, first token, prefix ratio), address similarity (Jaccard, overlap, n-gram and TF-IDF cosine, numeric
token overlap, length, missing flag), country match, source indicator, combined scores and candidate-list rank/gap.

## 7. Machine Learning Model
Random Forest, HistGradientBoosting and Logistic Regression were compared on validation F0.5.
Selected model: **{{MODEL}}**.

| Model | Precision | Recall | F0.5 | Threshold |
|---|---|---|---|---|
{{COMPARISON_ROWS}}

## 8. Threshold Optimization
Thresholds 0.05–0.95 (step 0.01) were evaluated on the validation set; the middle of the best F0.5 plateau is
used. Selected threshold: **{{THRESHOLD}}**.

## 9. Validation Results
Validation split: {{SPLIT}} (no entity appears in both train and validation).

* Precision: **{{PRECISION}}**
* Recall: **{{RECALL}}** (counts all true pairs, including those blocking missed)
* F0.5: **{{F05}}**
* TP / FP / FN: {{TP}} / {{FP}} / {{FN}}

## 10. Output Generation
Test run on {{S1_RECORDS}} Source 1 entities: {{CANDIDATES}} candidate pairs, {{FINAL_MATCHES}} final matches,
{{MATCHED}} entities with at least one match, {{UNMATCHED}} without. `validate_outputs.py` confirms row counts,
unique and valid IDs, no duplicates inside rows and that every match is inside its candidate list.

## 11. Visualization Dashboard
Flask + HTML/CSS/JS + Chart.js. KPI cards, match/unmatched chart, candidate vs match chart, pipeline view,
model metrics, candidate analysis, country distribution, searchable/filterable paginated results table,
refresh button, responsive layout. All values come from the output files via the API.

## 12. Technology Stack
Python, pandas, NumPy, SciPy, scikit-learn, Flask, Chart.js.

## 13. Limitations
* Only the first {{LIMIT}} records are used, as instructed.
* Transliteration is rule based and approximate; heavily different scripts/spellings can still be missed.
* Matches whose records lie outside the 10,000-record window cannot be recovered.
* Token-level blocking depends on shared name/address evidence; completely different names with no address
  overlap cannot be found.

## 14. Future Improvements
Learned transliteration, phonetic keys, pair-level gradient boosting with more address-structure features
(city/state parsing), cluster-level consistency between Source 2 and Source 3 matches, and using the full dataset.
