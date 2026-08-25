"""Final held-out evaluation: CVAT GT tracks -> TP/FP/FN -> TTFD."""

import logging
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.data.cvat_tracks import Track, first_visible_frame, visible_boxes_at
from src.metrics.distance import classify_bbox_distance_band
from src.metrics.matching import Box, match_predictions_to_gt
from src.metrics.task_evaluator import BAND_LABELS, GROUPS, DistanceConfig, empty_group_counts, group_metrics
from src.models.inference import FrameInference

logger = logging.getLogger(__name__)


def _percentile95(values: List[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) < 2:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def _aggregate_ttfd(pairs: List[Tuple[bool, Optional[float]]]) -> Dict:
    """pairs: (detected, ttfd_seconds_or_None) for every TTFD-eligible track in one group."""
    eligible = len(pairs)
    detected_values = [v for detected, v in pairs if detected]
    never_detected = eligible - len(detected_values)

    mean_v = statistics.mean(detected_values) if detected_values else None
    median_v = statistics.median(detected_values) if detected_values else None
    p95_v = _percentile95(detected_values)

    return {
        "time_to_first_detection": mean_v,  # headline metric, per spec
        "time_to_first_detection_mean": mean_v,
        "time_to_first_detection_median": median_v,
        "time_to_first_detection_p95": p95_v,
        "time_to_first_detection_skipped_reason": None if eligible else "no TTFD-eligible GT tracks in this group",
        "ttfd_eligible_tracks": eligible,
        "ttfd_detected_tracks": len(detected_values),
        "ttfd_never_detected_tracks": never_detected,
    }


def run_final_evaluation(
    frame_order: Sequence[Path],
    tracks: Dict[int, Track],
    predictions: Dict[str, FrameInference],
    matching_iou_threshold: float,
    distance_cfg: DistanceConfig,
    band_1_max_m: float,
    band_2_max_m: float,
    fps: float,
    exclude_frame0_tracks_from_ttfd: bool = True,
) -> Tuple[Dict, List[Dict], List[Dict]]:
    """Returns (metrics, frame_matches, ttfd_rows)."""
    counts = {g: empty_group_counts() for g in GROUPS}
    outside_range_counts = {"gt": 0, "fp": 0}
    frame_matches: List[Dict] = []
    track_first_tp_frame: Dict[int, int] = {}

    def _band(w_px: float, h_px: float, image_w: int):
        return classify_bbox_distance_band(
            w_px, h_px, image_w,
            fov_deg=distance_cfg.fov_deg,
            car_length_m=distance_cfg.car_length_m,
            car_width_m=distance_cfg.car_width_m,
            band_1_max_m=band_1_max_m,
            band_2_max_m=band_2_max_m,
        )

    for frame_idx, image_path in enumerate(frame_order):
        key = str(image_path)
        if key not in predictions:
            raise KeyError(f"No predictions found for eval frame {frame_idx}: {key}")
        frame_pred = predictions[key]
        image_w, image_h = frame_pred.image_width, frame_pred.image_height

        visible = visible_boxes_at(tracks, frame_idx)
        gt_track_ids = [tid for tid, _ in visible]
        gt_boxes = [Box(b.x1, b.y1, b.x2, b.y2) for _, b in visible]
        pred_boxes = [Box(b.x1, b.y1, b.x2, b.y2) for b in frame_pred.boxes]
        pred_confidences = [b.confidence for b in frame_pred.boxes]

        matches = match_predictions_to_gt(pred_boxes, pred_confidences, gt_boxes, matching_iou_threshold)

        frame_record = {"frame_idx": frame_idx, "image": image_path.name, "matches": []}

        for match in matches:
            if match.status in ("tp", "fn"):
                gt_box = gt_boxes[match.gt_index]
                w_px, h_px = gt_box.x2 - gt_box.x1, gt_box.y2 - gt_box.y1
                distance_m, band = _band(w_px, h_px, image_w)
                track_id = gt_track_ids[match.gt_index]
            else:  # fp -- no GT to anchor to, estimate from the prediction's own box
                pred_box = pred_boxes[match.pred_index]
                w_px, h_px = pred_box.x2 - pred_box.x1, pred_box.y2 - pred_box.y1
                distance_m, band = _band(w_px, h_px, image_w)
                track_id = None

            band_label = BAND_LABELS[band]
            counts["overall"][match.status] += 1
            if band_label != "outside_evaluation_range":
                counts[band_label][match.status] += 1
            elif match.status == "fn":
                outside_range_counts["gt"] += 1
            elif match.status == "fp":
                outside_range_counts["fp"] += 1

            if match.status == "tp":
                prev = track_first_tp_frame.get(track_id)
                if prev is None or frame_idx < prev:
                    track_first_tp_frame[track_id] = frame_idx

            frame_record["matches"].append(
                {
                    "status": match.status,
                    "gt_track_id": track_id,
                    "gt_bbox": list(gt_boxes[match.gt_index]) if match.gt_index is not None else None,
                    "pred_bbox": list(pred_boxes[match.pred_index]) if match.pred_index is not None else None,
                    "confidence": pred_confidences[match.pred_index] if match.pred_index is not None else None,
                    "iou": match.iou,
                    "estimated_distance_m": distance_m,
                    "distance_band": band_label,
                }
            )

        frame_matches.append(frame_record)

    # --- TTFD, per track ---
    ttfd_rows: List[Dict] = []
    ttfd_pairs_by_group: Dict[str, List[Tuple[bool, Optional[float]]]] = {g: [] for g in GROUPS}

    for track_id, track in tracks.items():
        fvf = first_visible_frame(track)
        if fvf is None:
            continue  # no visible boxes at all -- not a real eligible GT object

        excluded = exclude_frame0_tracks_from_ttfd and fvf == 0
        exclusion_reason = "track already visible at frame 0 (unknown true entry time)" if excluded else None

        fvf_box = track.boxes[fvf]
        w_px, h_px = fvf_box.x2 - fvf_box.x1, fvf_box.y2 - fvf_box.y1
        ref_pred = predictions[str(frame_order[fvf])]
        distance_m, band = _band(w_px, h_px, ref_pred.image_width)
        band_label = BAND_LABELS[band]

        first_detected_frame = track_first_tp_frame.get(track_id)
        detected = first_detected_frame is not None
        ttfd_frames = (first_detected_frame - fvf) if detected else None
        ttfd_seconds = (ttfd_frames / fps) if detected else None

        ttfd_rows.append(
            {
                "track_id": track_id,
                "first_visible_frame": fvf,
                "first_visible_distance_m": distance_m,
                "distance_band": band_label,
                "first_detected_frame": first_detected_frame,
                "detected": detected,
                "ttfd_frames": ttfd_frames,
                "ttfd_seconds": ttfd_seconds,
                "excluded_from_ttfd": excluded,
                "exclusion_reason": exclusion_reason,
            }
        )

        if not excluded:
            ttfd_pairs_by_group["overall"].append((detected, ttfd_seconds))
            if band_label in ("0_200m", "200_400m"):
                ttfd_pairs_by_group[band_label].append((detected, ttfd_seconds))

    n_frames = len(frame_order)
    metrics = {"eval": {"fps": fps, "n_frames": n_frames, "duration_seconds": n_frames / fps}}
    for g in GROUPS:
        gm = group_metrics(counts[g], n_frames, fps)
        gm.update(_aggregate_ttfd(ttfd_pairs_by_group[g]))
        metrics[g] = gm
    metrics["outside_evaluation_range_counts"] = outside_range_counts

    logger.info(
        "Final eval: %d frames | overall tp=%d fp=%d fn=%d DR=%s P=%s | %d TTFD-eligible tracks "
        "(%d detected, %d never detected, %d excluded frame-0)",
        n_frames, counts["overall"]["tp"], counts["overall"]["fp"], counts["overall"]["fn"],
        metrics["overall"]["detection_rate"], metrics["overall"]["precision"],
        metrics["overall"]["ttfd_eligible_tracks"], metrics["overall"]["ttfd_detected_tracks"],
        metrics["overall"]["ttfd_never_detected_tracks"],
        sum(1 for r in ttfd_rows if r["excluded_from_ttfd"]),
    )
    return metrics, frame_matches, ttfd_rows


def write_ttfd_csv(ttfd_rows: List[Dict], csv_path: Path) -> None:
    import csv

    fieldnames = [
        "track_id", "first_visible_frame", "first_visible_distance_m", "distance_band",
        "first_detected_frame", "detected", "ttfd_frames", "ttfd_seconds",
        "excluded_from_ttfd", "exclusion_reason",
    ]
    csv_path = Path(csv_path)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(ttfd_rows, key=lambda r: r["track_id"]):
            writer.writerow(row)
