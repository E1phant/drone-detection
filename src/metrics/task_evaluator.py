import logging
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence

from src.data.lovo_split import FrameRecord
from src.metrics.distance import classify_bbox_distance_band, yolo_bbox_to_pixel_dims
from src.metrics.matching import Box, match_predictions_to_gt
from src.models.inference import FrameInference

logger = logging.getLogger(__name__)

BAND_LABELS = {"band_1": "0_200m", "band_2": "200_400m", None: "outside_evaluation_range"}
GROUPS = ("overall", "0_200m", "200_400m")


class DistanceConfig(NamedTuple):
    fov_deg: float
    car_length_m: float
    car_width_m: float


def _parse_gt_line(line: str, image_width_px: int, image_height_px: int):
    parts = line.split()
    _class_id, xc, yc, w, h = parts[0], *map(float, parts[1:5])
    width_px, height_px = yolo_bbox_to_pixel_dims(w, h, image_width_px, image_height_px)
    x_center_px, y_center_px = xc * image_width_px, yc * image_height_px
    box = Box(
        x1=x_center_px - width_px / 2,
        y1=y_center_px - height_px / 2,
        x2=x_center_px + width_px / 2,
        y2=y_center_px + height_px / 2,
    )
    return box, width_px, height_px


def empty_group_counts() -> Dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0}


def group_metrics(counts: Dict[str, int], n_frames: int, fps: float) -> Dict:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]

    if tp + fn > 0:
        detection_rate, detection_rate_skipped_reason = tp / (tp + fn), None
    else:
        detection_rate, detection_rate_skipped_reason = None, "no GT boxes in this group"

    if tp + fp > 0:
        precision, precision_skipped_reason = tp / (tp + fp), None
    else:
        precision, precision_skipped_reason = None, "no predictions in this group"

    if n_frames > 0 and fps > 0:
        duration_min = (n_frames / fps) / 60.0
        false_alarms_per_min = fp / duration_min if duration_min > 0 else None
        false_alarms_skipped_reason = None
    else:
        false_alarms_per_min, false_alarms_skipped_reason = None, "no evaluated frames / invalid fps"

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "detection_rate": detection_rate,
        "detection_rate_skipped_reason": detection_rate_skipped_reason,
        "precision": precision,
        "precision_skipped_reason": precision_skipped_reason,
        "false_alarms_per_min": false_alarms_per_min,
        "false_alarms_per_min_skipped_reason": false_alarms_skipped_reason,
        "time_to_first_detection": None,
        "time_to_first_detection_skipped_reason": (
            "no GT track/object identity available in the gt_labels export -- TTFD requires "
            "temporal GT association across frames, which has not been prepared yet"
        ),
    }


def evaluate_fold(
    fold: str,
    held_out_video: Optional[str],
    val_records: Sequence[FrameRecord],
    predictions: Dict[str, FrameInference],
    matching_iou_threshold: float,
    distance_cfg: DistanceConfig,
    band_1_max_m: float,
    band_2_max_m: float,
    fps: float,
) -> tuple:
    """Returns (metrics_dict, detection_records) for one held-out-video fold."""
    counts = {"overall": empty_group_counts(), "0_200m": empty_group_counts(), "200_400m": empty_group_counts()}
    outside_range_counts = {"gt": 0, "fp": 0}
    detection_records: List[Dict] = []

    for record in val_records:
        key = str(record.image_path)
        if key not in predictions:
            raise KeyError(f"No predictions found for validation image {key}")
        frame_pred = predictions[key]
        image_w, image_h = frame_pred.image_width, frame_pred.image_height

        gt_lines = [l for l in record.label_path.read_text().splitlines() if l.strip()]
        gt_boxes, gt_pixel_dims = [], []
        for line in gt_lines:
            box, w_px, h_px = _parse_gt_line(line, image_w, image_h)
            gt_boxes.append(box)
            gt_pixel_dims.append((w_px, h_px))

        pred_boxes = [Box(b.x1, b.y1, b.x2, b.y2) for b in frame_pred.boxes]
        pred_confidences = [b.confidence for b in frame_pred.boxes]

        matches = match_predictions_to_gt(pred_boxes, pred_confidences, gt_boxes, matching_iou_threshold)

        for match in matches:
            if match.status in ("tp", "fn"):
                gt_index = match.gt_index
                w_px, h_px = gt_pixel_dims[gt_index]
                distance_m, band = classify_bbox_distance_band(
                    w_px, h_px, image_w,
                    fov_deg=distance_cfg.fov_deg,
                    car_length_m=distance_cfg.car_length_m,
                    car_width_m=distance_cfg.car_width_m,
                    band_1_max_m=band_1_max_m,
                    band_2_max_m=band_2_max_m,
                )
            else:  # fp: no GT to anchor to -- estimate from the prediction's own bbox
                pred_box = pred_boxes[match.pred_index]
                w_px, h_px = pred_box.x2 - pred_box.x1, pred_box.y2 - pred_box.y1
                distance_m, band = classify_bbox_distance_band(
                    w_px, h_px, image_w,
                    fov_deg=distance_cfg.fov_deg,
                    car_length_m=distance_cfg.car_length_m,
                    car_width_m=distance_cfg.car_width_m,
                    band_1_max_m=band_1_max_m,
                    band_2_max_m=band_2_max_m,
                )

            band_label = BAND_LABELS[band]
            counts["overall"][match.status] += 1
            if band_label != "outside_evaluation_range":
                counts[band_label][match.status] += 1
            elif match.status == "fn":
                outside_range_counts["gt"] += 1
            elif match.status == "fp":
                outside_range_counts["fp"] += 1

            detection_records.append(
                {
                    "image": record.image_path.name,
                    "status": match.status,
                    "bbox": list(pred_boxes[match.pred_index]) if match.pred_index is not None else list(gt_boxes[match.gt_index]),
                    "confidence": pred_confidences[match.pred_index] if match.pred_index is not None else None,
                    "class_id": 0,
                    "matched_gt_index": match.gt_index if match.status == "tp" else None,
                    "iou": match.iou,
                    "estimated_distance_m": distance_m,
                    "distance_band": band_label,
                }
            )

    n_frames = len(val_records)
    metrics = {
        "fold": fold,
        "held_out_video": held_out_video,
        "overall": group_metrics(counts["overall"], n_frames, fps),
        "0_200m": group_metrics(counts["0_200m"], n_frames, fps),
        "200_400m": group_metrics(counts["200_400m"], n_frames, fps),
        "outside_evaluation_range_counts": outside_range_counts,
        "n_val_frames": n_frames,
        "fps_used_for_false_alarms": fps,
    }
    logger.info(
        "Task evaluator [fold %s, held_out=%s]: overall tp=%d fp=%d fn=%d | 0-200m tp=%d fp=%d fn=%d "
        "| 200-400m tp=%d fp=%d fn=%d | outside_range gt=%d fp=%d",
        fold, held_out_video,
        counts["overall"]["tp"], counts["overall"]["fp"], counts["overall"]["fn"],
        counts["0_200m"]["tp"], counts["0_200m"]["fp"], counts["0_200m"]["fn"],
        counts["200_400m"]["tp"], counts["200_400m"]["fp"], counts["200_400m"]["fn"],
        outside_range_counts["gt"], outside_range_counts["fp"],
    )
    return metrics, detection_records


def flatten_task_metrics_for_wandb(metrics: Dict, standard_metrics: Dict) -> Dict:
    """Flatten this module's nested metrics dict into `task/<group>/<key>` W&B scalars.

    Only numeric fields are emitted (the `*_skipped_reason` strings stay in
    metrics.json for humans but aren't useful as W&B chart series); `None`
    values are dropped rather than logged as 0/NaN.
    """
    flat = {}
    for group in ("overall", "0_200m", "200_400m"):
        group_metrics = metrics.get(group) or {}
        for key in ("tp", "fp", "fn", "detection_rate", "precision", "false_alarms_per_min", "time_to_first_detection"):
            value = group_metrics.get(key)
            if value is not None:
                flat[f"task/{group}/{key}"] = value
    for key, value in (standard_metrics or {}).items():
        if value is not None:
            flat[f"standard_metrics/{key}"] = value
    return flat
