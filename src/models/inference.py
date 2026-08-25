import logging
import tempfile
from pathlib import Path
from typing import Dict, List, NamedTuple

from src.models.yolo_detector import YoloDetector

logger = logging.getLogger(__name__)


class PredictedBox(NamedTuple):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


class FrameInference(NamedTuple):
    image_path: Path
    image_width: int
    image_height: int
    boxes: List[PredictedBox]


def _run_full_frame_inference(
    detector: YoloDetector, image_paths: List[Path], conf: float, iou: float, max_det: int, batch: int = 8
) -> Dict[str, FrameInference]:
    image_paths = list(image_paths)
    resolved_by_key = {str(Path(p).resolve()): p for p in image_paths}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(str(Path(p).resolve()) for p in image_paths) + "\n")
        list_file = f.name

    try:
        results = detector.predict(
            source=list_file, conf=conf, iou=iou, max_det=max_det, stream=True, batch=batch, verbose=False,
        )

        predictions: Dict[str, FrameInference] = {}
        for result in results:
            key = str(Path(result.path).resolve())
            original_path = resolved_by_key.get(key)
            if original_path is None:
                raise RuntimeError(f"Unexpected result path not among the requested image_paths: {result.path}")

            height, width = result.orig_shape
            boxes = [
                PredictedBox(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    confidence=float(conf_score),
                    class_id=int(cls_id),
                )
                for xyxy, conf_score, cls_id in zip(
                    result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()
                )
            ]
            predictions[str(original_path)] = FrameInference(
                image_path=original_path, image_width=width, image_height=height, boxes=boxes
            )
    finally:
        Path(list_file).unlink(missing_ok=True)

    missing = set(resolved_by_key) - {str(Path(p.image_path).resolve()) for p in predictions.values()}
    if missing:
        raise RuntimeError(f"{len(missing)} requested image(s) got no inference result: {sorted(missing)[:5]}")

    return predictions


def run_inference(
    detector: YoloDetector,
    image_paths: List[Path],
    conf: float,
    iou: float,
    strategy: str = "full_frame",
    max_det: int = 300,
    batch: int = 8,
) -> Dict[str, FrameInference]:
    """Run detector inference over `image_paths`, keyed by (string) image path."""
    if strategy == "full_frame":
        return _run_full_frame_inference(detector, image_paths, conf, iou, max_det, batch=batch)
    raise NotImplementedError(
        f"inference.strategy={strategy!r} is not implemented yet."
    )


def filter_predictions_by_confidence(
    predictions: Dict[str, FrameInference], min_confidence: float
) -> Dict[str, FrameInference]:
    return {
        key: frame._replace(boxes=[b for b in frame.boxes if b.confidence >= min_confidence])
        for key, frame in predictions.items()
    }


def predictions_to_yolo_lines(boxes: List[PredictedBox], image_width: int, image_height: int) -> List[str]:
    """Format predicted boxes as `class x y w h conf` lines (YOLO + a trailing confidence column)."""
    lines = []
    for box in boxes:
        x1, y1 = max(0.0, min(box.x1, image_width)), max(0.0, min(box.y1, image_height))
        x2, y2 = max(0.0, min(box.x2, image_width)), max(0.0, min(box.y2, image_height))
        if x2 <= x1 or y2 <= y1:
            continue
        xc, yc = (x1 + x2) / 2.0 / image_width, (y1 + y2) / 2.0 / image_height
        w, h = (x2 - x1) / image_width, (y2 - y1) / image_height
        lines.append(f"{box.class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {box.confidence:.6f}")
    return lines
