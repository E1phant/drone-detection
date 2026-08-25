"""Parser for CVAT "for video" track-format XML exports (official eval GT)."""

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

logger = logging.getLogger(__name__)

FRAME_INDEX_RE = re.compile(r"_(\d+)\.[^.]+$")


class TrackBox(NamedTuple):
    frame_idx: int
    x1: float
    y1: float
    x2: float
    y2: float
    outside: bool
    keyframe: bool


class Track(NamedTuple):
    track_id: int
    label: str
    boxes: Dict[int, TrackBox]  # frame_idx -> box, visible and outside alike


class CvatMeta(NamedTuple):
    n_frames: int
    start_frame: int
    stop_frame: int
    labels: List[str]


def first_visible_frame(track: Track) -> Optional[int]:
    visible = [f for f, b in track.boxes.items() if not b.outside]
    return min(visible) if visible else None


def last_visible_frame(track: Track) -> Optional[int]:
    visible = [f for f, b in track.boxes.items() if not b.outside]
    return max(visible) if visible else None


def visible_boxes_at(tracks: Dict[int, Track], frame_idx: int) -> List[Tuple[int, TrackBox]]:
    """Every (track_id, box) visible (outside=0) at `frame_idx`."""
    result = []
    for track_id, track in tracks.items():
        box = track.boxes.get(frame_idx)
        if box is not None and not box.outside:
            result.append((track_id, box))
    return result


def _find_job_or_task(root: ET.Element) -> ET.Element:
    meta = root.find("meta")
    job = meta.find("job")
    if job is not None:
        return job
    task = meta.find("task")
    if task is not None:
        return task
    raise ValueError("CVAT XML <meta> has neither <job> nor <task> -- unrecognized export format")


def parse_cvat_tracks(xml_path: Path, expected_label: str = "vehicle") -> Tuple[Dict[int, Track], CvatMeta]:
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()
    job = _find_job_or_task(root)

    n_frames = int(job.findtext("size"))
    start_frame = int(job.findtext("start_frame"))
    stop_frame = int(job.findtext("stop_frame"))
    declared_labels = [l.findtext("name") for l in job.find("labels").findall("label")]

    tracks: Dict[int, Track] = {}
    seen_ids = set()
    skipped_labels = 0
    skipped_degenerate_boxes = 0
    frame_range_errors: List[str] = []

    for track_el in root.findall("track"):
        track_id = int(track_el.get("id"))
        label = track_el.get("label")

        if track_id in seen_ids:
            raise ValueError(f"Duplicate track id {track_id} in {xml_path}")
        seen_ids.add(track_id)

        if label != expected_label:
            logger.warning(
                "Track %d has label %r (expected %r) -- skipping this track entirely",
                track_id, label, expected_label,
            )
            skipped_labels += 1
            continue

        boxes: Dict[int, TrackBox] = {}
        for box_el in track_el.findall("box"):
            frame_idx = int(box_el.get("frame"))
            if not (start_frame <= frame_idx <= stop_frame):
                frame_range_errors.append(
                    f"track {track_id}: frame {frame_idx} outside declared range [{start_frame}, {stop_frame}]"
                )
                continue

            outside = box_el.get("outside") == "1"
            x1, y1 = float(box_el.get("xtl")), float(box_el.get("ytl"))
            x2, y2 = float(box_el.get("xbr")), float(box_el.get("ybr"))

            if not outside and (x2 <= x1 or y2 <= y1):
                logger.warning(
                    "Track %d frame %d: visible box has non-positive area (%.2f,%.2f)-(%.2f,%.2f) -- skipping this box",
                    track_id, frame_idx, x1, y1, x2, y2,
                )
                skipped_degenerate_boxes += 1
                continue

            boxes[frame_idx] = TrackBox(
                frame_idx=frame_idx, x1=x1, y1=y1, x2=x2, y2=y2,
                outside=outside, keyframe=box_el.get("keyframe") == "1",
            )

        tracks[track_id] = Track(track_id=track_id, label=label, boxes=boxes)

    if frame_range_errors:
        raise ValueError(
            f"{len(frame_range_errors)} box(es) had frame indices outside the declared range:\n"
            + "\n".join(frame_range_errors)
        )

    n_visible_boxes = sum(1 for t in tracks.values() for b in t.boxes.values() if not b.outside)
    starts = [first_visible_frame(t) for t in tracks.values()]
    starts = [s for s in starts if s is not None]
    n_start_at_0 = sum(1 for s in starts if s == 0)
    n_enter_later = sum(1 for s in starts if s > 0)

    logger.info(
        "Parsed CVAT tracks from %s: %d tracks (%d skipped: unexpected label), %d visible GT boxes, "
        "frame range [%d, %d], %d track(s) start at frame 0, %d enter later, %d degenerate box(es) skipped",
        xml_path, len(tracks), skipped_labels, n_visible_boxes, start_frame, stop_frame,
        n_start_at_0, n_enter_later, skipped_degenerate_boxes,
    )

    meta = CvatMeta(n_frames=n_frames, start_frame=start_frame, stop_frame=stop_frame, labels=declared_labels)
    return tracks, meta


def validate_frame_mapping(images_dir: Path, expected_n_frames: int, image_extensions=(".jpg", ".jpeg", ".png")) -> List[Path]:
    """Return eval image paths ordered by their true embedded frame index."""
    images_dir = Path(images_dir)
    image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in image_extensions]
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    indexed: List[Tuple[int, Path]] = []
    unparsable: List[str] = []
    for p in image_paths:
        match = FRAME_INDEX_RE.search(p.name)
        if match is None:
            unparsable.append(p.name)
            continue
        indexed.append((int(match.group(1)), p))

    if unparsable:
        raise ValueError(
            f"{len(unparsable)} image(s) in {images_dir} have no parsable trailing frame index:\n"
            + "\n".join(sorted(unparsable))
        )

    found_indices = sorted(idx for idx, _ in indexed)
    expected_indices = list(range(expected_n_frames))
    if found_indices != expected_indices:
        missing = sorted(set(expected_indices) - set(found_indices))
        extra = sorted(set(found_indices) - set(expected_indices))
        raise ValueError(
            f"Frame index mapping mismatch in {images_dir}: expected exactly {{0..{expected_n_frames - 1}}}, "
            f"got {len(found_indices)} images. Missing: {missing[:10]}{'...' if len(missing) > 10 else ''}; "
            f"unexpected/duplicate: {extra[:10]}{'...' if len(extra) > 10 else ''}"
        )

    return [p for _idx, p in sorted(indexed, key=lambda t: t[0])]


def tracks_to_yolo_labels(
    tracks: Dict[int, Track],
    n_frames: int,
    image_width: int,
    image_height: int,
    output_dir: Path,
    frame_stems: List[str],
    class_id: int = 0,
) -> Path:
    """Materialize one standard YOLO .txt per frame (all visible boxes, class 0)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in range(n_frames):
        lines = []
        for _track_id, box in visible_boxes_at(tracks, frame_idx):
            x1 = max(0.0, min(box.x1, image_width))
            y1 = max(0.0, min(box.y1, image_height))
            x2 = max(0.0, min(box.x2, image_width))
            y2 = max(0.0, min(box.y2, image_height))
            if x2 <= x1 or y2 <= y1:
                continue
            xc, yc = (x1 + x2) / 2.0 / image_width, (y1 + y2) / 2.0 / image_height
            w, h = (x2 - x1) / image_width, (y2 - y1) / image_height
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        label_path = output_dir / f"{frame_stems[frame_idx]}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    logger.info("Materialized YOLO GT labels for %d frames -> %s", n_frames, output_dir)
    return output_dir
