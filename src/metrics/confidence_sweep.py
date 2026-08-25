import csv
import logging
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.data.lovo_split import FrameRecord
from src.metrics.task_evaluator import GROUPS, DistanceConfig, evaluate_fold
from src.models.inference import FrameInference, filter_predictions_by_confidence

logger = logging.getLogger(__name__)

GROUP_METRIC_KEYS = ("tp", "fp", "fn", "detection_rate", "precision", "false_alarms_per_min")
FRAME_STAT_KEYS = (
    "mean_predictions_per_frame",
    "max_predictions_per_frame",
    "frames_hitting_max_det",
    "fraction_frames_hitting_max_det",
    "mean_fp_per_frame",
    "max_fp_per_frame",
)


def _frame_count_stats(predictions: Dict[str, FrameInference], max_det: int) -> Dict:
    counts = [len(frame.boxes) for frame in predictions.values()]
    n_frames = len(counts)
    hitting = sum(1 for c in counts if c >= max_det)
    return {
        "mean_predictions_per_frame": statistics.mean(counts) if counts else None,
        "max_predictions_per_frame": max(counts) if counts else None,
        "frames_hitting_max_det": hitting,
        "fraction_frames_hitting_max_det": hitting / n_frames if n_frames else None,
    }


def _fp_count_stats(detection_records: List[Dict], n_frames: int) -> Dict:
    fp_by_image = Counter(r["image"] for r in detection_records if r["status"] == "fp")
    zero_fp_frames = max(0, n_frames - len(fp_by_image))
    all_counts = list(fp_by_image.values()) + [0] * zero_fp_frames
    return {
        "mean_fp_per_frame": statistics.mean(all_counts) if all_counts else None,
        "max_fp_per_frame": max(all_counts) if all_counts else None,
    }


def run_confidence_sweep(
    fold: str,
    checkpoint: str,
    val_records: Sequence[FrameRecord],
    base_predictions: Dict[str, FrameInference],
    thresholds: Sequence[float],
    matching_iou_threshold: float,
    distance_cfg: DistanceConfig,
    band_1_max_m: float,
    band_2_max_m: float,
    fps: float,
    max_det: Optional[int] = None,
) -> Dict:
    sorted_thresholds = sorted(thresholds)
    n_frames = len(val_records)
    results = []
    for threshold in sorted_thresholds:
        filtered = filter_predictions_by_confidence(base_predictions, threshold)
        metrics, detection_records = evaluate_fold(
            fold=fold,
            held_out_video=None,  # not meaningful per-threshold; the fold-level metrics.json already records it
            val_records=val_records,
            predictions=filtered,
            matching_iou_threshold=matching_iou_threshold,
            distance_cfg=distance_cfg,
            band_1_max_m=band_1_max_m,
            band_2_max_m=band_2_max_m,
            fps=fps,
        )
        row = {"confidence": threshold, **{g: metrics[g] for g in GROUPS}}
        if max_det is not None:
            row.update(_frame_count_stats(filtered, max_det))
            row.update(_fp_count_stats(detection_records, n_frames))
        results.append(row)

    logger.info(
        "Confidence sweep [fold %s]: %d thresholds (%.3f-%.3f) evaluated from one inference pass -> %s",
        fold, len(sorted_thresholds), sorted_thresholds[0], sorted_thresholds[-1], checkpoint,
    )
    return {"fold": fold, "checkpoint": checkpoint, "max_det": max_det, "thresholds": results}


def write_confidence_sweep_csv(sweep_result: Dict, csv_path: Path) -> None:
    has_frame_stats = bool(sweep_result["thresholds"]) and FRAME_STAT_KEYS[0] in sweep_result["thresholds"][0]
    fieldnames = ["confidence"] + [f"{g}_{k}" for g in GROUPS for k in GROUP_METRIC_KEYS]
    if has_frame_stats:
        fieldnames += list(FRAME_STAT_KEYS)
    csv_path = Path(csv_path)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sweep_result["thresholds"]:
            flat = {"confidence": row["confidence"]}
            for g in GROUPS:
                for k in GROUP_METRIC_KEYS:
                    flat[f"{g}_{k}"] = row[g][k]
            if has_frame_stats:
                for k in FRAME_STAT_KEYS:
                    flat[k] = row.get(k)
            writer.writerow(flat)


def log_confidence_sweep_to_wandb(
    wandb_run, sweep_result: Dict, step: Optional[int] = None, key_prefix: str = "confidence_sweep"
) -> None:
    if wandb_run is None:
        return
    import wandb

    rows = sweep_result["thresholds"]
    confidences = [r["confidence"] for r in rows]
    has_frame_stats = bool(rows) and FRAME_STAT_KEYS[0] in rows[0]

    wide_columns = ["confidence"] + [f"{g}_{k}" for g in GROUPS for k in GROUP_METRIC_KEYS]
    if has_frame_stats:
        wide_columns += list(FRAME_STAT_KEYS)
    wide_table = wandb.Table(columns=wide_columns)
    for r in rows:
        row_values = [r["confidence"]] + [r[g][k] for g in GROUPS for k in GROUP_METRIC_KEYS]
        if has_frame_stats:
            row_values += [r.get(k) for k in FRAME_STAT_KEYS]
        wide_table.add_data(*row_values)

    def _series(metric_key):
        return [[r[g][metric_key] for r in rows] for g in GROUPS]

    log_payload = {
        f"{key_prefix}/table": wide_table,
        f"{key_prefix}/detection_rate_vs_confidence": wandb.plot.line_series(
            xs=confidences, ys=_series("detection_rate"), keys=list(GROUPS),
            title="Detection Rate vs Confidence", xname="confidence",
        ),
        f"{key_prefix}/precision_vs_confidence": wandb.plot.line_series(
            xs=confidences, ys=_series("precision"), keys=list(GROUPS),
            title="Precision vs Confidence", xname="confidence",
        ),
        f"{key_prefix}/false_alarms_per_min_vs_confidence": wandb.plot.line_series(
            xs=confidences, ys=_series("false_alarms_per_min"), keys=list(GROUPS),
            title="False Alarms/min vs Confidence", xname="confidence",
        ),
    }

    tradeoff_table = wandb.Table(columns=["confidence", "band", "detection_rate", "precision"])
    for r in rows:
        for g in GROUPS:
            tradeoff_table.add_data(r["confidence"], g, r[g]["detection_rate"], r[g]["precision"])
    log_payload[f"{key_prefix}/precision_vs_detection_rate"] = wandb.plot.line(
        tradeoff_table, x="detection_rate", y="precision", stroke="band",
        title="Precision vs Detection Rate (by confidence threshold)",
    )

    if step is not None:
        wandb_run.log(log_payload, step=step)
    else:
        wandb_run.log(log_payload)
