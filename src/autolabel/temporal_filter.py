import logging
import math
import re
import shutil
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

from src.autolabel.zero_shot import Detection, detections_to_yolo_lines
from src.utils.io import load_json, write_json

logger = logging.getLogger(__name__)

FRAME_STEM_RE = re.compile(r"^(?P<video_id>.+)_(?P<frame_idx>\d{6})$")


class FrameBox(NamedTuple):
    frame_stem: str
    frame_idx: int
    raw_index: int  # index into that frame's debug JSON "detections" list
    x_center: float
    y_center: float
    width: float
    height: float
    score: float
    label: str


class Track(NamedTuple):
    track_id: int
    video_id: str
    boxes: List[FrameBox]


def parse_frame_stem(frame_stem: str):
    """Split 'train_A_000045' -> ('train_A', 45); None if it doesn't match."""
    match = FRAME_STEM_RE.match(frame_stem)
    if not match:
        return None
    return match.group("video_id"), int(match.group("frame_idx"))


def _load_frame_boxes(debug_path: Path) -> tuple:
    """Load the final (post-NMS, post-containment) boxes for one frame's debug JSON.

    Returns (frame_boxes, image_width, image_height, class_agnostic detections
    in pixel space keyed by raw_index) for later YOLO re-export.
    """
    data = load_json(str(debug_path))
    width, height = data["image_width"], data["image_height"]
    frame_stem = Path(data["frame"]).stem
    parsed = parse_frame_stem(frame_stem)
    frame_idx = parsed[1] if parsed else 0

    boxes = []
    pixel_detections = {}
    for raw_index, det in enumerate(data["detections"]):
        if not det.get("is_final", det["survived_nms"] and not det["removed_by_containment"]):
            continue
        x1, y1, x2, y2 = det["bbox"]
        boxes.append(
            FrameBox(
                frame_stem=frame_stem,
                frame_idx=frame_idx,
                raw_index=raw_index,
                x_center=(x1 + x2) / 2.0 / width,
                y_center=(y1 + y2) / 2.0 / height,
                width=(x2 - x1) / width,
                height=(y2 - y1) / height,
                score=det["score"],
                label=det["label"],
            )
        )
        pixel_detections[raw_index] = Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=det["score"], label=det["label"])

    return boxes, width, height, pixel_detections


def build_tracks(
    frames: List[List[FrameBox]],
    video_id: str,
    max_center_distance: float,
    max_size_change: float,
) -> List[Track]:
    active: List[dict] = []
    finished: List[dict] = []
    next_id = 0

    for boxes in frames:
        used = set()
        matched_track_ids = set()

        for track in active:
            last = track["last"]
            best_idx, best_dist = None, None
            for i, box in enumerate(boxes):
                if i in used:
                    continue
                dist = math.hypot(box.x_center - last.x_center, box.y_center - last.y_center)
                if dist > max_center_distance:
                    continue
                w_change = abs(box.width - last.width) / max(last.width, 1e-6)
                h_change = abs(box.height - last.height) / max(last.height, 1e-6)
                if max(w_change, h_change) > max_size_change:
                    continue
                if best_dist is None or dist < best_dist:
                    best_dist, best_idx = dist, i
            if best_idx is not None:
                track["boxes"].append(boxes[best_idx])
                track["last"] = boxes[best_idx]
                used.add(best_idx)
                matched_track_ids.add(track["track_id"])

        still_active = []
        for track in active:
            (still_active if track["track_id"] in matched_track_ids else finished).append(track)
        active = still_active

        for i, box in enumerate(boxes):
            if i not in used:
                active.append({"track_id": next_id, "last": box, "boxes": [box]})
                next_id += 1

    finished.extend(active)
    return [Track(t["track_id"], video_id, t["boxes"]) for t in finished]


def identify_suspicious_tracks(
    tracks: List[Track], min_track_length: int, max_center_distance: float, max_confidence: float
) -> List[Track]:
    """Flag tracks that are long, near-perfectly stationary end-to-end, AND low-confidence.

    Stationary behaviour alone never triggers a flag -- confidence must also
    be low, so a confidently-detected stopped/parked vehicle is left alone.
    """
    suspicious = []
    for track in tracks:
        if len(track.boxes) < min_track_length:
            continue
        first, last = track.boxes[0], track.boxes[-1]
        total_displacement = math.hypot(last.x_center - first.x_center, last.y_center - first.y_center)
        if total_displacement > max_center_distance:
            continue
        mean_confidence = sum(b.score for b in track.boxes) / len(track.boxes)
        if mean_confidence > max_confidence:
            continue
        suspicious.append(track)
    return suspicious


def _track_summary(track: Track, suspicious: bool) -> Dict:
    xs = [b.x_center for b in track.boxes]
    ys = [b.y_center for b in track.boxes]
    ws = [b.width for b in track.boxes]
    hs = [b.height for b in track.boxes]
    scores = [b.score for b in track.boxes]
    first, last = track.boxes[0], track.boxes[-1]
    return {
        "track_id": track.track_id,
        "video_id": track.video_id,
        "length": len(track.boxes),
        "frame_start": first.frame_stem,
        "frame_end": last.frame_stem,
        "mean_center": [sum(xs) / len(xs), sum(ys) / len(ys)],
        "mean_size": [sum(ws) / len(ws), sum(hs) / len(hs)],
        "total_displacement": math.hypot(last.x_center - first.x_center, last.y_center - first.y_center),
        "mean_confidence": sum(scores) / len(scores),
        "max_confidence": max(scores),
        "suspicious": suspicious,
    }


def run_temporal_filter(
    labels_debug_dir: Path,
    raw_labels_dir: Path,
    clean_labels_dir: Path,
    target_video_id: str,
    class_id: int = 0,
    min_track_length: int = 6,
    max_center_distance: float = 0.005,
    max_size_change: float = 0.10,
    max_confidence: float = 0.50,
) -> Dict:
    raw_labels_dir = Path(raw_labels_dir)
    labels_debug_dir = Path(labels_debug_dir)
    clean_labels_dir = Path(clean_labels_dir)
    clean_labels_dir.mkdir(parents=True, exist_ok=True)

    raw_label_paths = sorted(raw_labels_dir.glob("*.txt"))
    if not raw_label_paths:
        logger.warning("No raw label files found in %s", raw_labels_dir)

    target_paths = []
    other_count = 0
    for label_path in raw_label_paths:
        parsed = parse_frame_stem(label_path.stem)
        if parsed and parsed[0] == target_video_id:
            target_paths.append((parsed[1], label_path))
        else:
            shutil.copy2(label_path, clean_labels_dir / label_path.name)
            other_count += 1
    target_paths.sort(key=lambda p: p[0])

    frames: List[List[FrameBox]] = []
    frame_meta = []  # (frame_idx, label_path, width, height, pixel_detections)
    missing_debug = 0
    for frame_idx, label_path in target_paths:
        debug_path = labels_debug_dir / f"{label_path.stem}.json"
        if not debug_path.exists():
            logger.warning(
                "No debug JSON for %s (expected %s); copying raw label unfiltered",
                label_path.name,
                debug_path,
            )
            shutil.copy2(label_path, clean_labels_dir / label_path.name)
            missing_debug += 1
            continue
        boxes, width, height, pixel_detections = _load_frame_boxes(debug_path)
        frames.append(boxes)
        frame_meta.append((frame_idx, label_path, width, height, pixel_detections))

    detections_analyzed = sum(len(b) for b in frames)

    tracks = build_tracks(
        frames, video_id=target_video_id,
        max_center_distance=max_center_distance, max_size_change=max_size_change,
    )
    multi_frame_tracks = [t for t in tracks if len(t.boxes) >= 2]
    suspicious_tracks = identify_suspicious_tracks(
        tracks, min_track_length=min_track_length,
        max_center_distance=max_center_distance, max_confidence=max_confidence,
    )
    suspicious_keys = {
        (b.frame_stem, b.raw_index) for track in suspicious_tracks for b in track.boxes
    }

    final_box_count = 0
    detections_marked_suspicious = 0
    for frame_idx, label_path, width, height, pixel_detections in frame_meta:
        kept = [
            det
            for raw_index, det in pixel_detections.items()
            if (label_path.stem, raw_index) not in suspicious_keys
        ]
        detections_marked_suspicious += len(pixel_detections) - len(kept)
        lines = detections_to_yolo_lines(kept, image_width=width, image_height=height, class_id=class_id)
        out_path = clean_labels_dir / label_path.name
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        final_box_count += len(lines)

    long_tracks_debug_path = labels_debug_dir / f"{target_video_id}_temporal_tracks.json"
    long_tracks = [t for t in tracks if len(t.boxes) >= min_track_length]
    write_json(
        str(long_tracks_debug_path),
        {
            "video_id": target_video_id,
            "min_track_length": min_track_length,
            "max_center_distance": max_center_distance,
            "max_size_change": max_size_change,
            "max_confidence": max_confidence,
            "tracks": [
                _track_summary(t, suspicious=t in suspicious_tracks) for t in long_tracks
            ],
        },
    )

    stats = {
        "video_id": target_video_id,
        "frames_analyzed": len(frames),
        "detections_analyzed": detections_analyzed,
        "temporal_sequences_found": len(multi_frame_tracks),
        "suspicious_sequences": len(suspicious_tracks),
        "detections_marked_suspicious": detections_marked_suspicious,
        "other_video_frames_copied_unchanged": other_count,
        "frames_missing_debug_json": missing_debug,
        "final_training_boxes": final_box_count,
    }
    logger.info(
        "Temporal filter [%s]: %d frames / %d detections analyzed -> %d sequences "
        "(>=2 frames), %d suspicious (>=%d frames, <=%.3f confidence, <=%.4f drift) "
        "-> %d detections dropped -> %d final boxes. Sequence diagnostics -> %s",
        target_video_id,
        stats["frames_analyzed"],
        stats["detections_analyzed"],
        stats["temporal_sequences_found"],
        stats["suspicious_sequences"],
        min_track_length,
        max_confidence,
        max_center_distance,
        stats["detections_marked_suspicious"],
        stats["final_training_boxes"],
        long_tracks_debug_path,
    )
    return stats
