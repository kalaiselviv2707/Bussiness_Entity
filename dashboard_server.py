"""Flask backend for the ENTITYRESOLVE AI dashboard.

Reads output/matching_results.tsv, output/candidate_pairs.tsv and
output/model_metrics.json and exposes them as JSON. Nothing is hardcoded:
every number shown in the dashboard comes from those files. Parsed files are
cached and reloaded automatically when their modification time changes.

Run:  python dashboard_server.py   ->   http://127.0.0.1:5000
"""
import json
import os
import sys
import threading
from collections import Counter
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config  # noqa: E402

DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"
MAX_PAGE_SIZE = 200

app = Flask(__name__, static_folder=None)
_lock = threading.Lock()
_cache = {}


class DataError(Exception):
    """A user-facing data problem (missing or malformed file)."""


def _split(value):
    return [x for x in value.split(",") if x] if value else []


def _read_tsv(path, header):
    if not path.exists():
        raise DataError(f"{path.name} not found. Run 'python src/predict.py' first.")
    rows = []
    with open(path, encoding="utf-8", newline="") as handle:
        lines = handle.read().splitlines()
    if not lines or lines[0].strip("\r") != "\t".join(header):
        raise DataError(f"{path.name} is malformed: expected header {header}.")
    for number, line in enumerate(lines[1:], start=2):
        parts = line.rstrip("\r").split("\t")
        if len(parts) == 1:
            parts.append("")
        if len(parts) != 2:
            raise DataError(f"{path.name} line {number} is malformed (expected 2 tab-separated columns).")
        rows.append((parts[0], parts[1]))
    if not rows:
        raise DataError(f"{path.name} is empty.")
    return rows


def _cached(key, path, loader):
    """Return loader(path), re-reading only when the file changed."""
    mtime = path.stat().st_mtime if path.exists() else None
    with _lock:
        entry = _cache.get(key)
        if entry and entry[0] == mtime and mtime is not None:
            return entry[1]
    value = loader(path)
    with _lock:
        _cache[key] = (mtime, value)
    return value


def load_matches():
    def loader(path):
        return [{"source1_entity_id": a, "matched": _split(b)}
                for a, b in _read_tsv(path, ["source1_entity_id", "matched_entity_ids"])]
    return _cached("matches", config.MATCHING_PATH, loader)


def load_candidates():
    def loader(path):
        return {a: _split(b) for a, b in _read_tsv(path, ["source1_entity_id", "candidate_entity_ids"])}
    return _cached("candidates", config.CANDIDATES_PATH, loader)


def load_metrics():
    path = config.METRICS_PATH
    if not path.exists():
        raise DataError("model_metrics.json not found. Run 'python src/train_model.py' first.")

    def loader(p):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DataError(f"model_metrics.json is not valid JSON: {exc}")
    return _cached("metrics", path, loader)


def _row(entry, candidate_count):
    return {"source1_entity_id": entry["source1_entity_id"],
            "matched_entity_ids": entry["matched"],
            "match_count": len(entry["matched"]),
            "candidate_count": candidate_count,
            "status": "Matched" if entry["matched"] else "Unmatched"}


def _filtered(status, query):
    matches, candidates = load_matches(), load_candidates()
    query = (query or "").strip().lower()
    rows = []
    for entry in matches:
        matched = bool(entry["matched"])
        if status == "matched" and not matched:
            continue
        if status == "unmatched" and matched:
            continue
        if query and query not in entry["source1_entity_id"].lower() \
                and not any(query in m.lower() for m in entry["matched"]):
            continue
        rows.append(_row(entry, len(candidates.get(entry["source1_entity_id"], []))))
    return rows


@app.errorhandler(DataError)
def handle_data_error(error):
    return jsonify({"error": str(error)}), 503


@app.errorhandler(404)
def handle_404(_):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(Exception)
def handle_unexpected(error):
    app.logger.exception("Unexpected error")
    return jsonify({"error": f"Internal error: {error}"}), 500


@app.get("/")
def index():
    return send_from_directory(DASHBOARD_DIR, "index.html")


@app.get("/static/<path:name>")
def assets(name):
    return send_from_directory(DASHBOARD_DIR, name)


@app.get("/api/dashboard")
def api_dashboard():
    matches, candidates = load_matches(), load_candidates()
    matched = sum(1 for m in matches if m["matched"])
    final_matches = sum(len(m["matched"]) for m in matches)
    candidate_pairs = sum(len(v) for v in candidates.values())
    match_size = Counter(min(len(m["matched"]), 5) for m in matches)
    try:
        prediction = load_metrics().get("prediction", {})
    except DataError:
        prediction = {}
    return jsonify({
        "total_source1": len(matches),
        "matched_entities": matched,
        "unmatched_entities": len(matches) - matched,
        "candidate_pairs": candidate_pairs,
        "final_matches": final_matches,
        "avg_candidates_per_entity": round(candidate_pairs / len(matches), 2),
        "matches_per_entity": {str(k): match_size.get(k, 0) for k in range(0, 6)},
        "by_country": prediction.get("by_country", {}),
        "record_limit": config.LIMIT,
    })


@app.get("/api/matches")
def api_matches():
    try:
        page = max(1, int(request.args.get("page", 1)))
        size = min(MAX_PAGE_SIZE, max(1, int(request.args.get("page_size", 25))))
    except ValueError:
        return jsonify({"error": "page and page_size must be integers"}), 400
    status = request.args.get("status", "all").lower()
    if status not in ("all", "matched", "unmatched"):
        return jsonify({"error": "status must be all, matched or unmatched"}), 400
    rows = _filtered(status, request.args.get("q"))
    start = (page - 1) * size
    return jsonify({"total": len(rows), "page": page, "page_size": size,
                    "pages": max(1, -(-len(rows) // size)), "rows": rows[start:start + size]})


@app.get("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    if len(query) < 2:
        return jsonify({"error": "Query must be at least 2 characters"}), 400
    rows = _filtered("all", query)
    return jsonify({"query": query, "total": len(rows), "rows": rows[:100]})


@app.get("/api/candidates")
def api_candidates():
    candidates, matches = load_candidates(), load_matches()
    counts = [len(v) for v in candidates.values()]
    buckets = [("0", 0, 0), ("1-10", 1, 10), ("11-25", 11, 25), ("26-50", 26, 50), ("51+", 51, 10 ** 9)]
    distribution = {label: sum(1 for c in counts if lo <= c <= hi) for label, lo, hi in buckets}
    final = sum(len(m["matched"]) for m in matches)
    total = sum(counts)
    top = sorted(candidates.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]
    return jsonify({
        "total_candidate_pairs": total, "final_matches": final,
        "rejected_candidates": total - final,
        "average": round(total / len(counts), 2), "max": max(counts), "min": min(counts),
        "entities_without_candidates": distribution["0"],
        "distribution": distribution,
        "top_entities": [{"source1_entity_id": k, "candidate_count": len(v)} for k, v in top],
    })


@app.get("/api/model-metrics")
def api_model_metrics():
    metrics = load_metrics()
    return jsonify({k: v for k, v in metrics.items() if k != "prediction"} | {
        "baseline": config.BASELINE})


@app.get("/api/health")
def api_health():
    files = {"matching_results": config.MATCHING_PATH, "candidate_pairs": config.CANDIDATES_PATH,
             "model_metrics": config.METRICS_PATH}
    status = {name: path.exists() for name, path in files.items()}
    status["all_ok"] = all(status.values())
    return jsonify(status)


@app.get("/api/feature-importance")
def api_feature_importance():
    import pickle
    import numpy as np
    model_path = config.MODEL_PATH
    if not model_path.exists():
        return jsonify({"error": "model.pkl not found"}), 503
    try:
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        model = saved.get("model")
        features = saved.get("features", [])
        importances = None
        # Try feature_importances_ directly
        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_).ravel().tolist()
        # Pipeline wrapping LogisticRegression
        elif hasattr(model, "steps"):
            inner = model.steps[-1][1]
            if hasattr(inner, "coef_"):
                importances = np.abs(np.asarray(inner.coef_).ravel()).tolist()
            elif hasattr(inner, "feature_importances_"):
                importances = np.asarray(inner.feature_importances_).ravel().tolist()
        # HistGradientBoosting: try _predictors tree-based importance
        if importances is None and hasattr(model, "_predictors"):
            try:
                n = len(features)
                imp = np.zeros(n)
                for stage in model._predictors:
                    for tree in stage:
                        nodes = tree.nodes
                        mask = nodes["is_leaf"] == 0
                        for idx, gain in zip(nodes["feature_idx"][mask], nodes["gain"][mask]):
                            if 0 <= idx < n:
                                imp[idx] += gain
                if imp.sum() > 0:
                    imp = imp / imp.sum()
                importances = imp.tolist()
            except Exception:
                importances = None
        if importances is None or len(importances) != len(features):
            return jsonify({"error": "Feature importances not available for this model type"}), 422
        pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        return jsonify({"features": [f for f, _ in pairs],
                        "importances": [round(v, 6) for _, v in pairs]})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/system-status")
def api_system_status():
    import pickle
    items = []
    # Output files
    for label, path in [("matching_results.tsv", config.MATCHING_PATH),
                        ("candidate_pairs.tsv",  config.CANDIDATES_PATH),
                        ("model_metrics.json",   config.METRICS_PATH)]:
        items.append({"name": label, "ok": path.exists(),
                      "detail": "Found" if path.exists() else "Missing"})
    # Model
    mp = config.MODEL_PATH
    if mp.exists():
        try:
            with open(mp, "rb") as f:
                saved = pickle.load(f)
            mname = saved.get("model_name", type(saved.get("model")).__name__)
            items.append({"name": "model.pkl", "ok": True, "detail": mname})
        except Exception as e:
            items.append({"name": "model.pkl", "ok": False, "detail": str(e)})
    else:
        items.append({"name": "model.pkl", "ok": False, "detail": "Missing"})
    return jsonify({"components": items, "all_ok": all(i["ok"] for i in items)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"ENTITYRESOLVE AI dashboard -> http://127.0.0.1:{port}  (output folder: {config.OUTPUT_DIR})")
    app.run(host="127.0.0.1", port=port, debug=False)
