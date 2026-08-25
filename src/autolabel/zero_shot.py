"""Zero-shot vehicle detection for YOLO-format pseudo-label generation."""

import logging
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import torch
from PIL import Image
from torchvision.ops import nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from src.utils.io import write_json

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class Detection(NamedTuple):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str


def nms_keep_indices(detections: List[Detection], iou_threshold: float) -> List[int]:
    """Return indices (into `detections`) surviving standard NMS, score-descending."""
    if not detections:
        return []

    boxes = torch.tensor(
        [[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=torch.float32
    )
    scores = torch.tensor([d.score for d in detections], dtype=torch.float32)
    return nms(boxes, scores, iou_threshold).tolist()


def containment_ratio(
    box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller_area = min(area_a, area_b)
    if smaller_area <= 0.0:
        return 0.0

    return inter_area / smaller_area


def apply_containment_filter(
    detections: List[Detection], containment_threshold: float
) -> Tuple[List[Detection], Set[int]]:
    if not detections:
        return [], set()

    order = sorted(range(len(detections)), key=lambda i: detections[i].score, reverse=True)
    removed: Set[int] = set()

    for pos, i in enumerate(order):
        if i in removed:
            continue
        box_i = (detections[i].x1, detections[i].y1, detections[i].x2, detections[i].y2)
        for j in order[pos + 1 :]:
            if j in removed:
                continue
            box_j = (detections[j].x1, detections[j].y1, detections[j].x2, detections[j].y2)
            if containment_ratio(box_i, box_j) >= containment_threshold:
                removed.add(j)

    kept = [detections[i] for i in range(len(detections)) if i not in removed]
    return kept, removed


def detections_to_yolo_lines(
    detections: List[Detection], image_width: int, image_height: int, class_id: int = 0
) -> List[str]:
    """Convert pixel-space detections to YOLO darknet label lines.

    Boxes are clipped to the image bounds; degenerate boxes are dropped.
    """
    lines = []
    for det in detections:
        x1 = max(0.0, min(det.x1, image_width))
        y1 = max(0.0, min(det.y1, image_height))
        x2 = max(0.0, min(det.x2, image_width))
        y2 = max(0.0, min(det.y2, image_height))
        if x2 <= x1 or y2 <= y1:
            continue

        x_center = (x1 + x2) / 2.0 / image_width
        y_center = (y1 + y2) / 2.0 / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return lines


class GroundingDinoAutoLabeler:

    def __init__(
        self,
        model_name: str = "IDEA-Research/grounding-dino-base",
        prompt: str = "vehicle.",
        box_threshold: float = 0.25,
        text_threshold: float = 0.2,
        nms_iou_threshold: float = 0.5,
        containment_threshold: float = 0.8,
        device: str = "cuda",
    ):
        self.prompt = prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.containment_threshold = containment_threshold

        self.device = device if torch.cuda.is_available() else "cpu"
        if self.device != device:
            logger.warning(
                "Requested device '%s' unavailable; falling back to '%s'",
                device,
                self.device,
            )

        logger.info("Loading zero-shot autolabeler '%s' on %s", model_name, self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, image: Image.Image) -> Tuple[List[Detection], List[Dict]]:
        """Run zero-shot detection + NMS + containment filtering on a single PIL image.

        Returns (final_detections, debug_records). `debug_records` covers every raw
        detection (pre-filtering) with flags for whether it survived NMS and whether
        it was subsequently dropped by containment filtering.
        """
        inputs = self.processor(
            images=image, text=self.prompt, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        raw_detections = [
            Detection(
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
                score=float(score),
                label=label,
            )
            for box, score, label in zip(
                results["boxes"], results["scores"], results["text_labels"]
            )
        ]

        nms_keep = nms_keep_indices(raw_detections, self.nms_iou_threshold)
        nms_keep_set = set(nms_keep)
        after_nms = [raw_detections[i] for i in nms_keep]

        _, containment_removed_local = apply_containment_filter(
            after_nms, self.containment_threshold
        )
        containment_removed_raw = {nms_keep[j] for j in containment_removed_local}

        final_detections = [
            raw_detections[i] for i in nms_keep if i not in containment_removed_raw
        ]

        debug_records = [
            {
                "bbox": [det.x1, det.y1, det.x2, det.y2],
                "score": det.score,
                "label": det.label,
                "survived_nms": idx in nms_keep_set,
                "removed_by_containment": idx in containment_removed_raw,
                "is_final": idx in nms_keep_set and idx not in containment_removed_raw,
            }
            for idx, det in enumerate(raw_detections)
        ]

        return final_detections, debug_records

    def label_image(
        self, image_path: Path, class_id: int = 0
    ) -> Tuple[List[str], List[Dict], int, Tuple[int, int]]:
        image = Image.open(image_path).convert("RGB")
        detections, debug_records = self.detect(image)
        lines = detections_to_yolo_lines(detections, image.width, image.height, class_id)
        return lines, debug_records, len(detections), (image.width, image.height)

    def label_directory(
        self,
        frames_dir: Path,
        labels_dir: Path,
        class_id: int = 0,
        labels_debug_dir: Optional[Path] = None,
        save_debug: bool = True,
    ) -> Dict[str, float]:
        """Generate one YOLO .txt pseudo-label file per frame in frames_dir.

        When `save_debug` is set, also writes one diagnostic JSON per frame
        (bbox/score/label/NMS survival/containment removal) to `labels_debug_dir`,
        without altering the YOLO .txt label format.
        """
        frames_dir = Path(frames_dir)
        labels_dir = Path(labels_dir)
        labels_dir.mkdir(parents=True, exist_ok=True)

        if save_debug:
            if labels_debug_dir is None:
                raise ValueError("labels_debug_dir must be set when save_debug=True")
            labels_debug_dir = Path(labels_debug_dir)
            labels_debug_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            logger.warning("No image frames found in %s", frames_dir)

        stats = {
            "frames_processed": 0,
            "total_raw_detections": 0,
            "total_after_nms": 0,
            "total_removed_by_nms": 0,
            "total_removed_by_containment": 0,
            "total_final_boxes": 0,
            "frames_with_zero_detections": 0,
        }

        for image_path in image_paths:
            lines, debug_records, final_count, (img_w, img_h) = self.label_image(
                image_path, class_id=class_id
            )
            label_path = labels_dir / f"{image_path.stem}.txt"
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

            raw_count = len(debug_records)
            after_nms_count = sum(1 for r in debug_records if r["survived_nms"])
            containment_removed_count = sum(
                1 for r in debug_records if r["removed_by_containment"]
            )

            stats["frames_processed"] += 1
            stats["total_raw_detections"] += raw_count
            stats["total_after_nms"] += after_nms_count
            stats["total_removed_by_nms"] += raw_count - after_nms_count
            stats["total_removed_by_containment"] += containment_removed_count
            stats["total_final_boxes"] += final_count
            if final_count == 0:
                stats["frames_with_zero_detections"] += 1

            if save_debug:
                debug_path = labels_debug_dir / f"{image_path.stem}.json"
                write_json(
                    str(debug_path),
                    {
                        "frame": image_path.name,
                        "image_width": img_w,
                        "image_height": img_h,
                        "prompt": self.prompt,
                        "detections": debug_records,
                    },
                )

            logger.info(
                "Labeled %s: %d raw -> %d after NMS -> %d final -> %s",
                image_path.name,
                raw_count,
                after_nms_count,
                final_count,
                label_path,
            )

        avg_final_boxes = (
            stats["total_final_boxes"] / stats["frames_processed"]
            if stats["frames_processed"]
            else 0.0
        )
        stats["avg_final_boxes_per_frame"] = round(avg_final_boxes, 3)

        logger.info(
            "Auto-labeling complete: %d frames | raw=%d -> after_nms=%d (removed_nms=%d) "
            "-> final=%d (removed_containment=%d) | avg %.2f boxes/frame | "
            "%d frame(s) with zero detections -> %s",
            stats["frames_processed"],
            stats["total_raw_detections"],
            stats["total_after_nms"],
            stats["total_removed_by_nms"],
            stats["total_final_boxes"],
            stats["total_removed_by_containment"],
            avg_final_boxes,
            stats["frames_with_zero_detections"],
            labels_dir,
        )
        return stats
