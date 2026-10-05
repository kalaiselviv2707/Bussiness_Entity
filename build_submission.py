"""Assemble Elite_Warriors_Submission.zip from the project and its generated outputs.

Run AFTER training + prediction + validation:
    python build_submission.py

It (1) validates the outputs, (2) fills Documentation_template.md with the real
numbers from output/model_metrics.json, (3) copies only the required files
(no dataset, model.pkl, caches or logs) and (4) creates and re-opens the ZIP.
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config  # noqa: E402
import validate_outputs  # noqa: E402

ROOT = Path(__file__).resolve().parent
NAME = "Elite_Warriors_Submission"
STAGE = ROOT / "_submission_build" / NAME
ZIP_PATH = ROOT / f"{NAME}.zip"
CODE_FILES = ["src/config.py", "src/preprocessing.py", "src/features.py", "src/train_model.py",
              "src/predict.py", "src/main.py", "src/validate_outputs.py",
              "dashboard/index.html", "dashboard/style.css", "dashboard/script.js",
              "dashboard_server.py", "README.md", "requirements.txt"]
OUTPUT_FILES = ["matching_results.tsv", "candidate_pairs.tsv", "model_metrics.json"]


def fill_documentation(metrics):
    text = (ROOT / "Documentation_template.md").read_text(encoding="utf-8")
    training, prediction = metrics.get("training", {}), metrics.get("prediction", {})
    comparison = "\n".join(
        f"| {r['model']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f0_5']:.4f} | {r['threshold']:.2f} |"
        for r in metrics.get("model_comparison", []))
    values = {
        "MODEL": metrics.get("model", "n/a"), "PRECISION": f"{metrics['precision']:.4f}",
        "RECALL": f"{metrics['recall']:.4f}", "F05": f"{metrics['f0_5']:.4f}",
        "THRESHOLD": f"{metrics['threshold']:.2f}", "TP": metrics.get("tp"), "FP": metrics.get("fp"),
        "FN": metrics.get("fn"), "COMPARISON_ROWS": comparison, "LIMIT": training.get("record_limit"),
        "BLOCKING_RECALL": training.get("blocking_recall"), "FEATURE_COUNT": training.get("feature_count"),
        "SPLIT": training.get("validation_split"), "TRAIN_CANDIDATES": training.get("candidate_pairs"),
        "TRUE_IN_SCOPE": training.get("true_pairs_in_scope"),
        "TRUE_OUTSIDE": training.get("true_pairs_outside_scope"),
        "S1_RECORDS": prediction.get("source1_records"), "MATCHED": prediction.get("matched_entities"),
        "UNMATCHED": (prediction.get("source1_records", 0) - prediction.get("matched_entities", 0)),
        "CANDIDATES": prediction.get("candidate_pairs"), "FINAL_MATCHES": prediction.get("final_matches"),
    }
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


def main():
    if not validate_outputs.validate():
        sys.exit("Output validation FAILED - fix the outputs before packaging.")
    if not config.METRICS_PATH.exists():
        sys.exit("model_metrics.json missing - run src/train_model.py and src/predict.py first.")
    metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    if "prediction" not in metrics:
        sys.exit("model_metrics.json has no prediction summary - run src/predict.py.")

    if STAGE.parent.exists():
        shutil.rmtree(STAGE.parent)
    code_dir = STAGE / "code" / "business_entity_resolution"
    (STAGE / "output").mkdir(parents=True)
    for name in OUTPUT_FILES:
        shutil.copy2(config.OUTPUT_DIR / name, STAGE / "output" / name)
    for relative in CODE_FILES:
        target = code_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    (STAGE / "Documentation_template.md").write_text(fill_documentation(metrics), encoding="utf-8")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                archive.write(path, Path(NAME) / path.relative_to(STAGE))
    shutil.rmtree(STAGE.parent)

    with zipfile.ZipFile(ZIP_PATH) as archive:       # verify it opens and is intact
        assert archive.testzip() is None, "ZIP is corrupt"
        names = archive.namelist()
    print(f"\nZIP created : {ZIP_PATH}\nZIP size    : {ZIP_PATH.stat().st_size / 1e6:.2f} MB\nFiles       : {len(names)}")
    for n in names:
        print("  ", n)
    bad = [n for n in names if "__pycache__" in n or n.endswith((".pkl", ".pyc", ".log"))]
    print("Unwanted files in ZIP:", bad or "none")


if __name__ == "__main__":
    main()
