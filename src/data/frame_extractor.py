"""OpenCV-based frame sampling for the aerial vehicle detection pipeline."""

import logging
from pathlib import Path
from typing import Dict, Iterable

import cv2

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_FPS = 30.0


class FrameExtractor:
    """Samples frames from MP4 clips at a fixed target FPS."""

    def __init__(self, fps: float = 2.0, image_ext: str = "jpg"):
        self.fps = fps
        self.image_ext = image_ext.lstrip(".")

    def extract_video(self, video_path: Path, output_dir: Path) -> int:
        """Sample frames from a single video and write them to output_dir.

        Returns the number of frames written.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not source_fps or source_fps <= 0:
            logger.warning(
                "Video %s reported invalid source FPS (%s); assuming %.1f",
                video_path.name,
                source_fps,
                DEFAULT_SOURCE_FPS,
            )
            source_fps = DEFAULT_SOURCE_FPS

        frame_interval = max(1, round(source_fps / self.fps))
        output_dir.mkdir(parents=True, exist_ok=True)

        frame_idx = 0
        saved_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                out_path = (
                    output_dir / f"{video_path.stem}_{frame_idx:06d}.{self.image_ext}"
                )
                cv2.imwrite(str(out_path), frame)
                saved_count += 1
            frame_idx += 1

        cap.release()
        logger.info(
            "Extracted %d frames from %s (source_fps=%.2f, interval=%d) -> %s",
            saved_count,
            video_path.name,
            source_fps,
            frame_interval,
            output_dir,
        )
        return saved_count

    def extract_dataset(
        self,
        raw_video_dir: Path,
        output_dir: Path,
        eval_video_names: Iterable[str] = (),
    ) -> Dict[str, int]:
        """Extract frames for every MP4 clip found in raw_video_dir.

        Clips whose stem is listed in eval_video_names are written under
        output_dir/eval/, all others under output_dir/train/.
        """
        raw_video_dir = Path(raw_video_dir)
        output_dir = Path(output_dir)
        eval_video_names = set(eval_video_names)

        video_paths = sorted(raw_video_dir.glob("*.mp4"))
        if not video_paths:
            logger.warning("No .mp4 files found in %s", raw_video_dir)

        counts = {"train": 0, "eval": 0}
        for video_path in video_paths:
            split = "eval" if video_path.stem in eval_video_names else "train"
            counts[split] += self.extract_video(video_path, output_dir / split)

        logger.info(
            "Frame extraction complete: %d train frames, %d eval frames",
            counts["train"],
            counts["eval"],
        )
        return counts
