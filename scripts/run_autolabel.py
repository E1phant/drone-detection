"""Generate zero-shot vehicle pseudo-labels for extracted training frames.

Usage:
    poetry run python scripts/run_autolabel.py
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="autolabel")
def main(cfg: DictConfig) -> None:
    logger.info("Auto-labeling with prompt: '%s'", cfg.autolabeler.prompt)
    labeler = hydra.utils.instantiate(cfg.autolabeler)
    stats = labeler.label_directory(
        frames_dir=Path(cfg.frames_dir),
        labels_dir=Path(cfg.labels_dir),
        class_id=cfg.class_id,
        labels_debug_dir=Path(cfg.labels_debug_dir),
        save_debug=cfg.save_debug,
    )
    logger.info("Auto-labeling finished: %s", stats)


if __name__ == "__main__":
    main()
