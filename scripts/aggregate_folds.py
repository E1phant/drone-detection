"""Cross-fold summary: aggregates outputs/fold_{A,B,C,D}/metrics.json into one table.

Usage:
    poetry run python scripts/aggregate_folds.py [--outputs-dir outputs] [--folds A B C D] [--no-wandb]

Per-fold rows are printed first (so a badly-generalizing video is visible on
its own row, not hidden behind an aggregate), followed by mean/std across
folds for every metric that's comparable across all of them.
"""

import argparse
import csv
import json
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from src.metrics.confidence_sweep import GROUP_METRIC_KEYS, GROUPS, log_confidence_sweep_to_wandb
from src.utils.io import write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TABLE_METRICS = [
    ("overall", "detection_rate", "Overall Detection Rate"),
    ("overall", "precision", "Overall Precision"),
    ("0_200m", "detection_rate", "0-200m Detection Rate"),
    ("0_200m", "precision", "0-200m Precision"),
    ("200_400m", "detection_rate", "200-400m Detection Rate"),
    ("200_400m", "precision", "200-400m Precision"),
    ("overall", "false_alarms_per_min", "False Alarms/min"),
    ("overall", "time_to_first_detection", "TTFD (s)"),
]


def load_fold_metrics(outputs_dir: Path, folds: List[str]) -> Dict[str, dict]:
    result = {}
    for fold in folds:
        path = outputs_dir / f"fold_{fold}" / "metrics.json"
        if not path.exists():
            logger.warning("Skipping fold %s: %s not found (train it first)", fold, path)
            continue
        result[fold] = json.loads(path.read_text())
    return result


def _get(metrics: dict, group: str, key: str) -> Optional[float]:
    if group not in metrics or not isinstance(metrics[group], dict):
        return None
    return metrics[group].get(key)


def build_summary(fold_metrics: Dict[str, dict]) -> dict:
    rows = []
    for fold, metrics in sorted(fold_metrics.items()):
        row = {"fold": fold, "held_out_video": metrics.get("held_out_video")}
        for group, key, _label in TABLE_METRICS:
            row[f"{group}/{key}"] = _get(metrics, group, key)
        row["map50"] = (metrics.get("standard_metrics") or {}).get("map50")
        rows.append(row)

    aggregates = {}
    for group, key, _label in TABLE_METRICS + [("standard_metrics", "map50", "mAP@0.5")]:
        metric_id = f"{group}/{key}"
        values = [r.get(metric_id) for r in rows if r.get(metric_id) is not None]
        if len(values) >= 2:
            aggregates[metric_id] = {"mean": statistics.mean(values), "std": statistics.stdev(values), "n": len(values)}
        elif len(values) == 1:
            aggregates[metric_id] = {"mean": values[0], "std": None, "n": 1}
        else:
            aggregates[metric_id] = {"mean": None, "std": None, "n": 0}

    return {"folds": rows, "aggregates": aggregates}


def print_table(summary: dict) -> None:
    header = ["Fold", "Held-out"] + [label for _, _, label in TABLE_METRICS] + ["mAP@0.5"]
    print(" | ".join(header))
    for row in summary["folds"]:
        cells = [row["fold"], str(row["held_out_video"])]
        for group, key, _label in TABLE_METRICS:
            v = row.get(f"{group}/{key}")
            cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "n/a")
        v = row.get("map50")
        cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "n/a")
        print(" | ".join(cells))

    print("\nAcross folds (mean +/- std, n):")
    for group, key, label in TABLE_METRICS + [("standard_metrics", "map50", "mAP@0.5")]:
        agg = summary["aggregates"][f"{group}/{key}"]
        if agg["mean"] is None:
            print(f"  {label}: n/a")
        elif agg["std"] is None:
            print(f"  {label}: {agg['mean']:.3f} (n=1)")
        else:
            print(f"  {label}: {agg['mean']:.3f} +/- {agg['std']:.3f} (n={agg['n']})")


def load_fold_sweeps(outputs_dir: Path, folds: List[str]) -> Dict[str, dict]:
    result = {}
    for fold in folds:
        path = outputs_dir / f"fold_{fold}" / "confidence_sweep.json"
        if not path.exists():
            logger.warning("Skipping fold %s: %s not found (train it with confidence_sweep.enabled=true)", fold, path)
            continue
        result[fold] = json.loads(path.read_text())
    return result


def aggregate_confidence_sweeps(fold_sweeps: Dict[str, dict]) -> dict:
    """Group each fold's confidence_sweep.json by threshold, then mean/std across folds.

    This aggregate -- not any single fold's sweep -- is what the final
    inference confidence threshold should be chosen from.
    """
    # "Worse" direction differs by metric: lower detection_rate/precision is
    # worse, higher false_alarms_per_min is worse.
    WORST_DIRECTION = {"detection_rate": min, "precision": min, "false_alarms_per_min": max}

    by_threshold: Dict[float, dict] = {}
    for fold, sweep in fold_sweeps.items():
        for entry in sweep["thresholds"]:
            conf = entry["confidence"]
            bucket = by_threshold.setdefault(conf, {g: {k: [] for k in GROUP_METRIC_KEYS} for g in GROUPS})
            for g in GROUPS:
                for k in GROUP_METRIC_KEYS:
                    v = entry[g].get(k)
                    if v is not None:
                        bucket[g][k].append((fold, v))

    rows = []
    for conf in sorted(by_threshold):
        row = {"confidence": conf}
        for g in GROUPS:
            for k in GROUP_METRIC_KEYS:
                pairs = by_threshold[conf][g][k]
                values = [v for _fold, v in pairs]
                if len(values) >= 2:
                    row[f"{g}_{k}_mean"] = statistics.mean(values)
                    row[f"{g}_{k}_std"] = statistics.stdev(values)
                elif len(values) == 1:
                    row[f"{g}_{k}_mean"] = values[0]
                    row[f"{g}_{k}_std"] = 0.0
                else:
                    row[f"{g}_{k}_mean"] = None
                    row[f"{g}_{k}_std"] = None
                row[f"{g}_{k}_n"] = len(values)
                if k in WORST_DIRECTION and pairs:
                    worst_fold, worst_value = WORST_DIRECTION[k](pairs, key=lambda p: p[1])
                    row[f"{g}_{k}_worst"] = worst_value
                    row[f"{g}_{k}_worst_fold"] = worst_fold
                elif k in WORST_DIRECTION:
                    row[f"{g}_{k}_worst"] = None
                    row[f"{g}_{k}_worst_fold"] = None
        rows.append(row)

    return {"folds": sorted(fold_sweeps.keys()), "thresholds": rows}


def write_confidence_sweep_aggregate_csv(aggregate: dict, csv_path: Path) -> None:
    fieldnames = ["confidence"]
    for g in GROUPS:
        for k in GROUP_METRIC_KEYS:
            stats = ("mean", "std", "n", "worst", "worst_fold") if k in ("detection_rate", "precision", "false_alarms_per_min") else ("mean", "std", "n")
            fieldnames += [f"{g}_{k}_{stat}" for stat in stats]
    csv_path = Path(csv_path)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregate["thresholds"]:
            writer.writerow(row)


def print_confidence_sweep_table(aggregate: dict) -> None:
    print(f"\nCross-fold confidence sweep (mean across {len(aggregate['folds'])} fold(s): {aggregate['folds']}):")
    header = ["conf", "mean overall DR", "worst DR (fold)", "mean overall P", "worst P (fold)", "mean overall FP/min", "mean 0-200m DR", "mean 200-400m DR"]
    print(" | ".join(header))
    for row in aggregate["thresholds"]:
        cells = [f"{row['confidence']:.3f}"]
        for key in ("overall_detection_rate_mean",):
            v = row.get(key)
            cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "n/a")
        v, vf = row.get("overall_detection_rate_worst"), row.get("overall_detection_rate_worst_fold")
        cells.append(f"{v:.3f} ({vf})" if isinstance(v, (int, float)) else "n/a")
        v = row.get("overall_precision_mean")
        cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "n/a")
        v, vf = row.get("overall_precision_worst"), row.get("overall_precision_worst_fold")
        cells.append(f"{v:.3f} ({vf})" if isinstance(v, (int, float)) else "n/a")
        for key in ("overall_false_alarms_per_min_mean", "0_200m_detection_rate_mean", "200_400m_detection_rate_mean"):
            v = row.get(key)
            cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "n/a")
        print(" | ".join(cells))


def _flat_aggregate_to_pseudo_sweep(aggregate: dict) -> dict:
    """Reshape the flat mean/std/n aggregate into run_confidence_sweep's nested
    shape (means only) so log_confidence_sweep_to_wandb can be reused as-is."""
    rows = [
        {"confidence": row["confidence"], **{g: {k: row.get(f"{g}_{k}_mean") for k in GROUP_METRIC_KEYS} for g in GROUPS}}
        for row in aggregate["thresholds"]
    ]
    return {"fold": "cross-fold-mean", "checkpoint": None, "thresholds": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--folds", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--no-wandb", action="store_true", help="Skip logging the summary to W&B")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    fold_metrics = load_fold_metrics(outputs_dir, args.folds)
    if not fold_metrics:
        logger.error("No fold metrics.json files found under %s -- train the folds first", outputs_dir)
        return

    summary = build_summary(fold_metrics)
    print_table(summary)

    summary_path = outputs_dir / "cross_fold_summary.json"
    write_json(str(summary_path), summary)
    logger.info("Cross-fold summary written to %s", summary_path)

    fold_sweeps = load_fold_sweeps(outputs_dir, args.folds)
    sweep_aggregate = None
    if fold_sweeps:
        sweep_aggregate = aggregate_confidence_sweeps(fold_sweeps)
        print_confidence_sweep_table(sweep_aggregate)
        write_json(str(outputs_dir / "cross_fold_confidence_sweep.json"), sweep_aggregate)
        write_confidence_sweep_aggregate_csv(sweep_aggregate, outputs_dir / "cross_fold_confidence_sweep.csv")
        logger.info("Cross-fold confidence sweep written to %s", outputs_dir / "cross_fold_confidence_sweep.json")
    else:
        logger.warning(
            "No confidence_sweep.json files found under %s -- skipping cross-fold confidence sweep aggregation "
            "(train folds with confidence_sweep.enabled=true, the current default, to produce them)", outputs_dir
        )

    if not args.no_wandb:
        try:
            import wandb

            run = wandb.init(project="drone-vehicle-detection", name="vehicle-detector-cross-fold-summary")
            table = wandb.Table(
                columns=["fold", "held_out_video"] + [f"{g}/{k}" for g, k, _ in TABLE_METRICS] + ["map50"],
                data=[
                    [row["fold"], row["held_out_video"]]
                    + [row.get(f"{g}/{k}") for g, k, _ in TABLE_METRICS]
                    + [row.get("map50")]
                    for row in summary["folds"]
                ],
            )
            run.log({"cross_fold_summary": table})
            run.log({f"aggregate/{k}/mean": v["mean"] for k, v in summary["aggregates"].items() if v["mean"] is not None})
            if sweep_aggregate is not None:
                log_confidence_sweep_to_wandb(
                    run, _flat_aggregate_to_pseudo_sweep(sweep_aggregate), key_prefix="cross_fold_confidence_sweep"
                )
            run.finish()
        except Exception:
            logger.warning("W&B logging of the cross-fold summary failed -- continuing without it", exc_info=True)


if __name__ == "__main__":
    main()
