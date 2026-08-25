import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YoloDetector:
    """Wraps `ultralytics.YOLO` behind a small, Hydra-friendly interface."""

    def __init__(self, checkpoint: str = "yolov8s.pt", image_size: int = 1920):
        self.checkpoint = checkpoint
        self.image_size = image_size
        logger.info("Loading YOLO checkpoint '%s' (image_size=%d)", checkpoint, image_size)
        self.model = YOLO(checkpoint)

    def train(self, **kwargs: Any):
        return self.model.train(imgsz=self.image_size, **kwargs)

    def val(self, **kwargs: Any):
        return self.model.val(imgsz=self.image_size, **kwargs)

    def predict(self, **kwargs: Any):
        return self.model.predict(imgsz=self.image_size, **kwargs)

    @property
    def save_dir(self) -> Path:
        trainer = getattr(self.model, "trainer", None)
        if trainer is None:
            raise RuntimeError("No trainer found -- call .train() first")
        return Path(trainer.save_dir).resolve()

    @property
    def best_checkpoint_path(self) -> Path:
        """Path to `weights/best.pt` for the just-completed training run, as Ultralytics itself wrote it."""
        trainer = getattr(self.model, "trainer", None)
        if trainer is None:
            raise RuntimeError("No trainer found -- call .train() first")
        return Path(trainer.best).resolve()

    @classmethod
    def from_checkpoint(cls, checkpoint: str, image_size: int) -> "YoloDetector":
        """Load a specific checkpoint file (e.g. a fold's `weights/best.pt`) at the same image size."""
        return cls(checkpoint=checkpoint, image_size=image_size)


def extract_standard_metrics(results) -> Dict[str, Optional[float]]:
    """Pull mAP50/mAP50-95 out of an Ultralytics `DetMetrics`-like results object."""
    results_dict = getattr(results, "results_dict", None) or {}
    return {
        "map50": results_dict.get("metrics/mAP50(B)"),
        "map50_95": results_dict.get("metrics/mAP50-95(B)"),
    }
