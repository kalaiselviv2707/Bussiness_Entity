"""Train the entity-resolution classifier and choose the F0.5-optimal threshold.

Steps: load first LIMIT records -> preprocess -> blocking -> features ->
grouped train/validation split (by Source 1 entity, no leakage) -> compare
lightweight models -> tune threshold for F0.5 -> save model + metrics.
"""
import json
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from features import FEATURE_COLUMNS, Representation, compute_pair_features, generate_candidates
from preprocessing import combine_sources, load_tsv, preprocess_dataframe


def load_training_data(limit):
    source1 = load_tsv(config.TRAIN_DIR / "train_source1.tsv", limit)
    source2 = load_tsv(config.TRAIN_DIR / "train_source2.tsv", limit)
    source3 = load_tsv(config.TRAIN_DIR / "train_source3.tsv", limit)
    truth = load_tsv(config.TRAIN_DIR / "train_ground_truth.tsv")
    return source1, source2, source3, truth


def build_truth_keys(truth_df, source1, source23):
    """Encoded (left_index * n_right + right_index) keys of ground-truth pairs that
    are inside the loaded records, plus how many true pairs are outside them."""
    left_index = pd.Series(np.arange(len(source1)), index=source1["entity_id"])
    right_index = pd.Series(np.arange(len(source23)), index=source23["entity_id"])
    right_index = right_index[~right_index.index.duplicated()]
    truth_df = truth_df[truth_df["source1_entity_id"].isin(left_index.index)]
    keys, outside = [], 0
    for s1_id, matched in zip(truth_df["source1_entity_id"], truth_df["matched_entity_ids"]):
        for entity_id in filter(None, (x.strip() for x in str(matched).split(","))):
            if entity_id in right_index.index:
                keys.append(int(left_index[s1_id]) * len(source23) + int(right_index[entity_id]))
            else:
                outside += 1
    return np.unique(np.array(keys, dtype=np.int64)), outside


def f_beta(precision, recall, beta=config.F_BETA):
    denominator = beta ** 2 * precision + recall
    return (1 + beta ** 2) * precision * recall / denominator if denominator > 0 else 0.0


def evaluate(y_true, probability, threshold, total_true):
    """Precision / recall / F0.5 where ``total_true`` counts ALL true pairs (also the
    ones blocking missed), so recall is not inflated by the candidate generator."""
    predicted = probability >= threshold
    tp = int(np.sum(predicted & (y_true == 1)))
    fp = int(np.sum(predicted & (y_true == 0)))
    fn = int(total_true - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_true if total_true else 0.0
    return {"threshold": float(threshold), "precision": precision, "recall": recall,
            "f0_5": f_beta(precision, recall), "tp": tp, "fp": fp, "fn": fn}


def best_threshold(y_true, probability, total_true):
    grid = np.arange(config.THRESHOLD_GRID_START, config.THRESHOLD_GRID_END, config.THRESHOLD_GRID_STEP)
    results = [evaluate(y_true, probability, t, total_true) for t in grid]
    top = max(r["f0_5"] for r in results)
    plateau = [r for r in results if r["f0_5"] >= top - 1e-9]
    return plateau[len(plateau) // 2]      # middle of the best plateau = most stable threshold


def candidate_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=16, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=config.RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, random_state=config.RANDOM_STATE),
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced")),
    }


def main():
    started = time.time()
    limit = config.LIMIT
    print("=" * 60)
    print("BUSINESS ENTITY RESOLUTION - TRAINING")
    print(f"Processing the first {limit:,} records of each training source")
    print("=" * 60)

    source1, source2, source3, truth_df = load_training_data(limit)
    print(f"Source 1: {len(source1):,} | Source 2: {len(source2):,} | Source 3: {len(source3):,} "
          f"| ground-truth rows (all): {len(truth_df):,}")
    source1 = preprocess_dataframe(source1)
    source23 = preprocess_dataframe(combine_sources(source2, source3))

    # ---- ground truth inside the loaded records ----
    true_keys, outside = build_truth_keys(truth_df, source1, source23)
    n_right = len(source23)
    print(f"True pairs inside the first {limit:,} records: {len(true_keys):,} "
          f"(true pairs pointing outside them, unreachable: {outside:,})")
    if len(true_keys) == 0:
        raise SystemExit("ERROR: no ground-truth pair lies inside the loaded records; cannot train.")

    # ---- blocking ----
    print("\nGenerating candidates (blocking)...")
    rep = Representation(source1, source23)
    nat_i, nat_j = generate_candidates(rep)
    natural_keys = nat_i * n_right + nat_j
    covered = np.isin(true_keys, natural_keys)
    print(f"Natural candidate pairs: {len(natural_keys):,} "
          f"({len(natural_keys) / len(source1):.1f} per Source 1 entity)")
    print(f"Blocking recall on reachable true pairs: {covered.mean():.4f}")

    # true matches missed by blocking are injected for TRAINING only
    injected_keys = true_keys[~covered]
    pool_keys = np.concatenate([natural_keys, injected_keys])
    pool_i, pool_j = pool_keys // n_right, pool_keys % n_right
    natural_mask = np.arange(len(pool_keys)) < len(natural_keys)
    labels = np.isin(pool_keys, true_keys).astype(np.int8)

    print("Computing features...")
    X = compute_pair_features(rep, pool_i, pool_j, natural_mask)
    print(f"Training pool: {len(X):,} pairs, {int(labels.sum()):,} positives, {X.shape[1]} features")

    # ---- grouped split by Source 1 entity ----
    rng = np.random.default_rng(config.RANDOM_STATE)
    left_ids = np.unique(pool_i)
    val_left = set(rng.choice(left_ids, size=int(len(left_ids) * config.VALIDATION_FRACTION), replace=False))
    is_val = np.isin(pool_i, list(val_left))
    train_idx = np.flatnonzero(~is_val)
    val_idx = np.flatnonzero(is_val & natural_mask)          # validate on natural candidates only
    total_true_val = int(np.isin(true_keys // n_right, list(val_left)).sum())
    print(f"Train pairs: {len(train_idx):,} | validation pairs: {len(val_idx):,} "
          f"| validation true pairs: {total_true_val:,}")

    # ---- model comparison ----
    comparison, fitted = [], {}
    for name, model in candidate_models().items():
        model.fit(X.iloc[train_idx], labels[train_idx])
        proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        result = best_threshold(labels[val_idx], proba, total_true_val)
        result["model"] = name
        comparison.append(result)
        fitted[name] = model
        print(f"  {name:<22} F0.5={result['f0_5']:.4f}  P={result['precision']:.4f}  "
              f"R={result['recall']:.4f}  thr={result['threshold']:.2f}")

    best = max(comparison, key=lambda r: r["f0_5"])
    chosen = best["model"]
    print("\n" + "=" * 60)
    print(f"SELECTED MODEL (highest validation F0.5): {chosen}")
    print("=" * 60)
    print(f"Best threshold : {best['threshold']:.2f}")
    print(f"Precision      : {best['precision']:.4f}")
    print(f"Recall         : {best['recall']:.4f}")
    print(f"F0.5           : {best['f0_5']:.4f}")
    print(f"TP / FP / FN   : {best['tp']} / {best['fp']} / {best['fn']}")
    base = config.BASELINE
    print(f"Previous baseline: P={base['precision']} R={base['recall']} F0.5={base['f0_5']} "
          f"thr={base['threshold']} (random pair split, 5 features)")

    with open(config.MODEL_PATH, "wb") as handle:
        pickle.dump({"model": fitted[chosen], "model_name": chosen,
                     "threshold": best["threshold"], "features": FEATURE_COLUMNS}, handle)

    metrics = {
        "precision": round(best["precision"], 4), "recall": round(best["recall"], 4),
        "f0_5": round(best["f0_5"], 4), "threshold": round(best["threshold"], 2),
        "tp": best["tp"], "fp": best["fp"], "fn": best["fn"],
        "model": chosen,
        "model_comparison": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                             for r in comparison],
        "training": {
            "record_limit": limit, "source1_records": len(source1), "source23_records": n_right,
            "true_pairs_in_scope": int(len(true_keys)), "true_pairs_outside_scope": int(outside),
            "blocking_recall": round(float(covered.mean()), 4),
            "candidate_pairs": int(len(natural_keys)), "training_pairs": int(len(X)),
            "positive_pairs": int(labels.sum()), "feature_count": len(FEATURE_COLUMNS),
            "validation_split": "grouped by Source 1 entity, "
                                f"{int(config.VALIDATION_FRACTION * 100)}% of entities",
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        },
    }
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nModel saved   : {config.MODEL_PATH}")
    print(f"Metrics saved : {config.METRICS_PATH}")
    print(f"Training time : {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
