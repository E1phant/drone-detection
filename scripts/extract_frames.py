"""Extract frames from raw aerial video clips into the train/eval image dataset.

Usage:
    poetry run python scripts/extract_frames.py
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.data.frame_extractor import FrameExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../configs/data", config_name="pexels_aerial"
)
def main(cfg: DictConfig) -> None:
    logger.info("Starting frame extraction at %.1f FPS", cfg.fps)
    extractor = FrameExtractor(fps=cfg.fps, image_ext=cfg.image_ext)
    counts = extractor.extract_dataset(
        raw_video_dir=Path(cfg.raw_video_dir),
        output_dir=Path(cfg.output_dir),
        eval_video_names=cfg.eval_video_names,
    )
    logger.info("Frame extraction finished: %s", counts)


if __name__ == "__main__":
    main()
