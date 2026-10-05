"""Generate matching_results.tsv and candidate_pairs.tsv for the test data."""
import json
import pickle
import time

import numpy as np
import pandas as pd

import config
from features import Representation, compute_pair_features, generate_candidates
from preprocessing import combine_sources, load_tsv, preprocess_dataframe


def write_outputs(source1, source23, pair_i, pair_j, probability, threshold):
    """Write both TSV files. Every Source 1 entity appears exactly once; every final
    match is taken from (and therefore contained in) its candidate list."""
    frame = pd.DataFrame({"i": pair_i, "j": pair_j, "p": probability})
    frame = frame.sort_values(["i", "p"], ascending=[True, False])
    right_ids = source23["entity_id"].to_numpy()

    candidates = {i: [] for i in range(len(source1))}
    matches = {i: [] for i in range(len(source1))}
    for i, j, p in zip(frame["i"], frame["j"], frame["p"]):
        entity_id = right_ids[j]
        if entity_id not in candidates[i]:
            candidates[i].append(entity_id)
            if p >= threshold:
                matches[i].append(entity_id)

    ids = source1["entity_id"].tolist()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"source1_entity_id": ids,
                  "matched_entity_ids": [",".join(matches[i]) for i in range(len(ids))]}
                 ).to_csv(config.MATCHING_PATH, sep="\t", index=False)
    pd.DataFrame({"source1_entity_id": ids,
                  "candidate_entity_ids": [",".join(candidates[i]) for i in range(len(ids))]}
                 ).to_csv(config.CANDIDATES_PATH, sep="\t", index=False)
    return candidates, matches


def update_metrics_file(source1, candidates, matches):
    """Add a small summary of this prediction run to model_metrics.json (for the dashboard)."""
    countries = source1["country"].replace("", "Unknown")
    matched_flags = pd.Series([bool(matches[i]) for i in range(len(source1))])
    by_country = {}
    for country, group in matched_flags.groupby(countries.to_numpy()):
        by_country[country] = {"total": int(len(group)), "matched": int(group.sum())}
    summary = {"source1_records": len(source1),
               "matched_entities": int(matched_flags.sum()),
               "candidate_pairs": int(sum(len(v) for v in candidates.values())),
               "final_matches": int(sum(len(v) for v in matches.values())),
               "by_country": by_country}
    metrics = {}
    if config.METRICS_PATH.exists():
        try:
            metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metrics = {}
    metrics["prediction"] = summary
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return summary


def main():
    started = time.time()
    limit = config.LIMIT
    print("=" * 60)
    print("BUSINESS ENTITY RESOLUTION - PREDICTION")
    print(f"Processing the first {limit:,} records of each test source")
    print("=" * 60)

    if not config.MODEL_PATH.exists():
        raise SystemExit(f"ERROR: model not found at {config.MODEL_PATH}. Run train_model.py first.")
    with open(config.MODEL_PATH, "rb") as handle:
        saved = pickle.load(handle)
    model, threshold, columns = saved["model"], saved["threshold"], saved["features"]
    print(f"Model: {saved.get('model_name', type(model).__name__)} | threshold: {threshold:.2f}")

    source1 = load_tsv(config.TEST_DIR / "test_source1.tsv", limit)
    source2 = load_tsv(config.TEST_DIR / "test_source2.tsv", limit)
    source3 = load_tsv(config.TEST_DIR / "test_source3.tsv", limit)
    print(f"Source 1: {len(source1):,} | Source 2: {len(source2):,} | Source 3: {len(source3):,}")

    source1_p = preprocess_dataframe(source1)
    source23 = preprocess_dataframe(combine_sources(source2, source3))

    rep = Representation(source1_p, source23)
    pair_i, pair_j = generate_candidates(rep)
    print(f"Candidate pairs: {len(pair_i):,} ({len(pair_i) / max(len(source1), 1):.1f} per Source 1 entity)")

    features = compute_pair_features(rep, pair_i, pair_j)[columns]
    probability = model.predict_proba(features)[:, 1]          # one batch call

    candidates, matches = write_outputs(source1_p, source23, pair_i, pair_j, probability, threshold)
    summary = update_metrics_file(source1_p, candidates, matches)
    print(f"Source 1 entities with a match: {summary['matched_entities']:,} "
          f"| without: {summary['source1_records'] - summary['matched_entities']:,} "
          f"| total final matches: {summary['final_matches']:,}")
    print(f"Wrote {config.MATCHING_PATH}\nWrote {config.CANDIDATES_PATH}")
    print(f"Prediction time: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
