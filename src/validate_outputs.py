"""Validate the generated output files against the challenge format rules."""
import sys

import pandas as pd

import config


def _read(path, columns, errors):
    if not path.exists():
        errors.append(f"{path.name} does not exist")
        return None
    try:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    except Exception as exc:  # malformed TSV
        errors.append(f"{path.name} could not be parsed as TSV: {exc}")
        return None
    if list(df.columns) != columns:
        errors.append(f"{path.name} columns are {list(df.columns)}, expected {columns}")
        return None
    return df


def _split(value):
    return [x for x in value.split(",") if x] if value else []


def validate():
    errors = []
    matching = _read(config.MATCHING_PATH, ["source1_entity_id", "matched_entity_ids"], errors)
    candidates = _read(config.CANDIDATES_PATH, ["source1_entity_id", "candidate_entity_ids"], errors)

    # reference IDs from the test data (skipped with a warning if the dataset is absent)
    valid_source1 = valid_right = None
    try:
        s1 = pd.read_csv(config.TEST_DIR / "test_source1.tsv", sep="\t", dtype=str,
                         keep_default_na=False).head(config.LIMIT)
        s2 = pd.read_csv(config.TEST_DIR / "test_source2.tsv", sep="\t", dtype=str,
                         keep_default_na=False).head(config.LIMIT)
        s3 = pd.read_csv(config.TEST_DIR / "test_source3.tsv", sep="\t", dtype=str,
                         keep_default_na=False).head(config.LIMIT)
        valid_source1 = list(s1["entity_id"])
        valid_right = set(s2["entity_id"]) | set(s3["entity_id"])
    except FileNotFoundError:
        print(f"WARNING: test dataset not found in {config.TEST_DIR}; ID-validity checks skipped.")

    stats = {"matching_rows": 0, "candidate_rows": 0, "dup_s1_matching": 0, "dup_s1_candidates": 0,
             "invalid_s1": 0, "invalid_ids": 0, "dup_in_match_rows": 0, "dup_in_cand_rows": 0,
             "outside_candidates": 0, "empty": 0, "non_empty": 0}

    if matching is not None:
        stats["matching_rows"] = len(matching)
        stats["dup_s1_matching"] = int(matching["source1_entity_id"].duplicated().sum())
        stats["empty"] = int((matching["matched_entity_ids"] == "").sum())
        stats["non_empty"] = len(matching) - stats["empty"]
    if candidates is not None:
        stats["candidate_rows"] = len(candidates)
        stats["dup_s1_candidates"] = int(candidates["source1_entity_id"].duplicated().sum())

    if matching is not None and candidates is not None:
        cand_map = dict(zip(candidates["source1_entity_id"], candidates["candidate_entity_ids"]))
        for s1_id, matched in zip(matching["source1_entity_id"], matching["matched_entity_ids"]):
            ids = _split(matched)
            if len(ids) != len(set(ids)):
                stats["dup_in_match_rows"] += 1
            allowed = set(_split(cand_map.get(s1_id, "")))
            stats["outside_candidates"] += sum(1 for x in ids if x not in allowed)
            if valid_right is not None:
                stats["invalid_ids"] += sum(1 for x in ids if x not in valid_right)
        for cand in candidates["candidate_entity_ids"]:
            ids = _split(cand)
            if len(ids) != len(set(ids)):
                stats["dup_in_cand_rows"] += 1
            if valid_right is not None:
                stats["invalid_ids"] += sum(1 for x in ids if x not in valid_right)
        if set(matching["source1_entity_id"]) != set(candidates["source1_entity_id"]):
            errors.append("Source 1 IDs differ between matching_results.tsv and candidate_pairs.tsv")

    if valid_source1 is not None and matching is not None:
        if list(matching["source1_entity_id"]) != valid_source1:
            wrong = set(matching["source1_entity_id"]) ^ set(valid_source1)
            stats["invalid_s1"] = len(wrong)
            errors.append(f"Source 1 IDs do not match the test file ({len(wrong)} differences or wrong order)")
        expected = len(valid_source1)
    else:
        expected = config.LIMIT
    if stats["matching_rows"] != expected or stats["candidate_rows"] != expected:
        errors.append(f"Expected {expected} rows, found matching={stats['matching_rows']}, "
                      f"candidates={stats['candidate_rows']}")
    for key, message in [("dup_s1_matching", "duplicate Source 1 IDs in matching_results.tsv"),
                         ("dup_s1_candidates", "duplicate Source 1 IDs in candidate_pairs.tsv"),
                         ("dup_in_match_rows", "rows with duplicate matched IDs"),
                         ("dup_in_cand_rows", "rows with duplicate candidate IDs"),
                         ("invalid_ids", "invalid Source 2/3 IDs"),
                         ("outside_candidates", "matches outside candidate lists")]:
        if stats[key]:
            errors.append(f"{stats[key]} {message}")

    print("=" * 50)
    print("OUTPUT VALIDATION")
    print("=" * 50)
    print(f"Output folder: {config.OUTPUT_DIR}")
    print(f"Matching rows: {stats['matching_rows']}")
    print(f"Candidate rows: {stats['candidate_rows']}")
    print(f"Duplicate Source 1 IDs: {stats['dup_s1_matching'] + stats['dup_s1_candidates']}")
    print(f"Invalid Source 1 IDs: {stats['invalid_s1']}")
    print(f"Invalid matches/candidates: {stats['invalid_ids']}")
    print(f"Duplicate IDs inside rows: {stats['dup_in_match_rows'] + stats['dup_in_cand_rows']}")
    print(f"Matches outside candidates: {stats['outside_candidates']}")
    print(f"Empty matches: {stats['empty']} | Non-empty matches: {stats['non_empty']}")
    for error in errors:
        print(f"ERROR: {error}")
    passed = not errors
    print(f"Validation: {'PASSED' if passed else 'FAILED'}")
    print("=" * 50)
    return passed


if __name__ == "__main__":
    sys.exit(0 if validate() else 1)
