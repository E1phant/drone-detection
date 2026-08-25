import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

from src.autolabel.temporal_filter import parse_frame_stem
from src.utils.io import write_json

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class FrameRecord(NamedTuple):
    image_path: Path
    label_path: Path
    video_id: str
    frame_idx: int
    box_count: int


class FoldStats(NamedTuple):
    fold: str
    held_out_video: Optional[str]
    n_train_images: int
    n_val_images: int
    n_train_boxes: int
    n_val_boxes: int
    train_video_distribution: Dict[str, int]
    val_video_distribution: Dict[str, int]


def ensure_labels_symlink(train_dataset_dir: Path, labels_target: str = "gt_labels") -> Path:
    """Idempotently create `<train_dataset_dir>/labels -> <labels_target>`.

    Ultralytics only ever looks for a sibling "labels" directory next to
    "images"; this symlink lets it find our `gt_labels` without copying or
    renaming anything. Refuses to touch a `labels/` that already exists and
    isn't exactly this symlink -- it may be the user's own data.
    """
    train_dataset_dir = Path(train_dataset_dir)
    labels_link = train_dataset_dir / "labels"
    target = train_dataset_dir / labels_target

    if not target.is_dir():
        raise FileNotFoundError(f"Expected label directory not found: {target}")

    if labels_link.is_symlink():
        if labels_link.resolve() != target.resolve():
            raise FileExistsError(
                f"{labels_link} already exists as a symlink to {labels_link.resolve()}, "
                f"not {target.resolve()} -- refusing to overwrite it."
            )
        return labels_link

    if labels_link.exists():
        raise FileExistsError(
            f"{labels_link} already exists and is not a symlink -- refusing to overwrite it. "
            f"Remove or rename it manually if it is safe to replace with a symlink to {labels_target}/."
        )

    labels_link.symlink_to(labels_target, target_is_directory=True)
    logger.info("Created label symlink: %s -> %s", labels_link, labels_target)
    return labels_link


def discover_frames(images_dir: Path, labels_dir: Path) -> List[FrameRecord]:
    """Discover every frame's source video from its filename and validate it has a label file."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    records = []
    errors = []
    for image_path in image_paths:
        parsed = parse_frame_stem(image_path.stem)
        if parsed is None:
            errors.append(f"{image_path.name}: filename does not match '<video_id>_<frame_idx>'")
            continue
        video_id, frame_idx = parsed

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            errors.append(f"{image_path.name}: missing label file {label_path}")
            continue

        lines = [l for l in label_path.read_text().splitlines() if l.strip()]
        records.append(
            FrameRecord(
                image_path=image_path,
                label_path=label_path,
                video_id=video_id,
                frame_idx=frame_idx,
                box_count=len(lines),
            )
        )

    if errors:
        raise ValueError(
            f"{len(errors)} frame(s) failed discovery in {images_dir}:\n" + "\n".join(errors)
        )

    return records


def _video_distribution(records: List[FrameRecord]) -> Dict[str, int]:
    return dict(sorted(Counter(r.video_id for r in records).items()))


def resolve_video_id(fold: str, known_videos: List[str]) -> str:
    if fold in known_videos:
        return fold
    suffix_matches = [v for v in known_videos if v == fold or v.endswith(f"_{fold}")]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        raise ValueError(f"fold={fold!r} is ambiguous among the videos found in the dataset: {suffix_matches}")
    raise ValueError(f"fold={fold!r} does not match any video found in the dataset: {known_videos}")


def build_fold(
    frames: List[FrameRecord],
    video_id: str,
    splits_dir: Path,
    class_names: Dict[int, str],
    fold_label: Optional[str] = None,
) -> FoldStats:
    
    fold_label = fold_label or video_id
    known_videos = sorted({r.video_id for r in frames})
    if video_id != "all" and video_id not in known_videos:
        raise ValueError(f"video_id={video_id!r} is not among the videos found in the dataset: {known_videos}")

    if video_id == "all":
        train_records = list(frames)
        val_records = list(frames)
        held_out_video = None
    else:
        train_records = [r for r in frames if r.video_id != video_id]
        val_records = [r for r in frames if r.video_id == video_id]
        held_out_video = video_id

        train_videos = {r.video_id for r in train_records}
        val_videos = {r.video_id for r in val_records}
        if val_videos != {video_id}:
            raise AssertionError(f"video_id={video_id}: validation set contains videos {val_videos}, expected only {{{video_id}}}")
        if video_id in train_videos:
            raise AssertionError(f"video_id={video_id}: training set unexpectedly contains the held-out video")
        train_paths = {r.image_path for r in train_records}
        val_paths = {r.image_path for r in val_records}
        overlap = train_paths & val_paths
        if overlap:
            raise AssertionError(f"video_id={video_id}: {len(overlap)} image(s) appear in both train and val: {sorted(overlap)[:5]}...")

    train_records = sorted(train_records, key=lambda r: r.image_path)
    val_records = sorted(val_records, key=lambda r: r.image_path)

    fold_dir = Path(splits_dir) / f"fold_{fold_label}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_txt = fold_dir / "train.txt"
    val_txt = fold_dir / "val.txt"
    train_txt.write_text("\n".join(str(r.image_path.resolve()) for r in train_records) + "\n")
    val_txt.write_text("\n".join(str(r.image_path.resolve()) for r in val_records) + "\n")

    dataset_yaml = fold_dir / "dataset.yaml"
    names_block = "\n".join(f"  {k}: {v}" for k, v in sorted(class_names.items()))
    dataset_yaml.write_text(
        f"train: {train_txt.resolve()}\n"
        f"val: {val_txt.resolve()}\n"
        f"names:\n{names_block}\n"
    )

    stats = FoldStats(
        fold=fold_label,
        held_out_video=held_out_video,
        n_train_images=len(train_records),
        n_val_images=len(val_records),
        n_train_boxes=sum(r.box_count for r in train_records),
        n_val_boxes=sum(r.box_count for r in val_records),
        train_video_distribution=_video_distribution(train_records),
        val_video_distribution=_video_distribution(val_records),
    )
    write_json(str(fold_dir / "split_stats.json"), stats._asdict())
    logger.info(
        "Fold %s: held_out=%s | train=%d images / %d boxes %s | val=%d images / %d boxes %s -> %s",
        fold_label,
        held_out_video,
        stats.n_train_images,
        stats.n_train_boxes,
        stats.train_video_distribution,
        stats.n_val_images,
        stats.n_val_boxes,
        stats.val_video_distribution,
        dataset_yaml,
    )
    return stats
