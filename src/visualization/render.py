"""Ground-truth vs prediction visualization for the official final evaluation."""

import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

GT_MATCHED_COLOR = (0, 200, 0)  # green -- GT with a correct matching prediction (TP)
GT_MISSED_COLOR = (0, 140, 255)  # orange -- GT with no matching prediction (FN)
PRED_TP_COLOR = (255, 255, 0)  # cyan -- prediction matched to a GT (TP)
PRED_FP_COLOR = (0, 0, 255)  # red -- unmatched prediction (FP)

_LEGEND_ENTRIES = (
    ("GT (detected)", GT_MATCHED_COLOR),
    ("GT (missed / FN)", GT_MISSED_COLOR),
    ("Prediction (TP)", PRED_TP_COLOR),
    ("Prediction (FP)", PRED_FP_COLOR),
)


def _draw_label(img: np.ndarray, text: str, x: int, y: int, color, font_scale: float = 1.3, thickness: int = 3) -> None:
    """Text on a filled background rectangle in `color`, for legibility over any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    y = max(y, th + baseline + 4)
    cv2.rectangle(img, (x, y - th - baseline - 4), (x + tw + 8, y + baseline), color, -1)
    text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
    cv2.putText(img, text, (x + 4, y - baseline), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_legend(img: np.ndarray) -> np.ndarray:
    """Small fixed legend in the top-left corner explaining the four box colors."""
    pad, line_h, swatch = 16, 44, 30
    box_h = pad * 2 + line_h * len(_LEGEND_ENTRIES)
    box_w = 430
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    for i, (label, color) in enumerate(_LEGEND_ENTRIES):
        y = pad + i * line_h
        cv2.rectangle(img, (pad, y), (pad + swatch, y + swatch), color, -1)
        cv2.putText(img, label, (pad + swatch + 12, y + swatch - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def draw_frame(image_path: Path, frame_record: Dict, draw_gt: bool = True, legend: bool = False) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image {image_path}")

    for match in frame_record["matches"]:
        status = match["status"]
        if draw_gt and match["gt_bbox"] is not None:
            x1, y1, x2, y2 = (int(v) for v in match["gt_bbox"])
            color = GT_MATCHED_COLOR if status == "tp" else GT_MISSED_COLOR
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 6)
            _draw_label(img, f"GT #{match['gt_track_id']}" if status == "tp" else "GT (missed)", x1, y1 - 12, color)
        if match["pred_bbox"] is not None:
            x1, y1, x2, y2 = (int(v) for v in match["pred_bbox"])
            color = PRED_TP_COLOR if status == "tp" else PRED_FP_COLOR
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 6)
            conf = match["confidence"]
            label = f"{'TP' if status == 'tp' else 'FP'} {conf:.2f}" if conf is not None else status.upper()
            _draw_label(img, label, x1, y2 + 46, color)

    if legend:
        img = draw_legend(img)

    return img


def select_representative_frames(frame_matches: List[Dict], ttfd_rows: List[Dict]) -> Dict[str, int]:
    def count_status(frame_record: Dict, status: str) -> int:
        return sum(1 for m in frame_record["matches"] if m["status"] == status)

    most_tp = max(frame_matches, key=lambda fr: count_status(fr, "tp"))
    most_fp = max(frame_matches, key=lambda fr: count_status(fr, "fp"))
    most_fn = max(frame_matches, key=lambda fr: count_status(fr, "fn"))

    selection = {
        "most_true_positives": most_tp["frame_idx"],
        "most_false_positives": most_fp["frame_idx"],
        "most_false_negatives": most_fn["frame_idx"],
    }

    farthest_frame_idx, farthest_dist = None, -1.0
    for fr in frame_matches:
        for m in fr["matches"]:
            if m["status"] == "tp" and m["estimated_distance_m"] is not None and m["estimated_distance_m"] > farthest_dist:
                farthest_dist = m["estimated_distance_m"]
                farthest_frame_idx = fr["frame_idx"]
    if farthest_frame_idx is not None:
        selection["farthest_correct_detection"] = farthest_frame_idx

    detected_rows = [r for r in ttfd_rows if r["detected"] and not r["excluded_from_ttfd"]]
    if detected_rows:
        slowest = max(detected_rows, key=lambda r: r["ttfd_seconds"])
        selection["slowest_detection"] = slowest["first_detected_frame"]

    return selection


def save_representative_frames(frame_order: List[Path], frame_matches: List[Dict], selection: Dict[str, int], output_dir: Path) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for label, frame_idx in selection.items():
        img = draw_frame(frame_order[frame_idx], frame_matches[frame_idx], legend=True)
        out_path = output_dir / f"{label}_frame{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), img)
        saved.append(out_path)
    logger.info("Saved %d representative frame(s) -> %s", len(saved), output_dir)
    return saved


def render_eval_video(frame_order: List[Path], frame_matches: List[Dict], output_path: Path, fps: float, draw_gt: bool = True) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_img = cv2.imread(str(frame_order[0]))
    if first_img is None:
        raise FileNotFoundError(f"Could not read first eval frame {frame_order[0]}")
    height, width = first_img.shape[:2]

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for frame_record, image_path in zip(frame_matches, frame_order):
            writer.write(draw_frame(image_path, frame_record, draw_gt=draw_gt))
    finally:
        writer.release()

    logger.info("Rendered eval video (%d frames @ %.1f fps) -> %s", len(frame_order), fps, output_path)
    return output_path
