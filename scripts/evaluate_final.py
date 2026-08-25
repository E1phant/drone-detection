"""Official final held-out evaluation: CVAT GT tracks -> TP/FP/FN/TTFD/mAP.

Usage:
    poetry run python scripts/evaluate_final.py
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
from PIL import Image

from src.data.cvat_tracks import parse_cvat_tracks, tracks_to_yolo_labels, validate_frame_mapping
from src.metrics.final_eval import DistanceConfig, run_final_evaluation, write_ttfd_csv
from src.models.inference import predictions_to_yolo_lines, run_inference
from src.models.yolo_detector import extract_standard_metrics
from src.utils.io import write_json
from src.utils.wandb_helpers import ensure_ultralytics_wandb_enabled, resolve_entity, safe_wandb_finish, safe_wandb_init
from src.visualization.render import render_eval_video, save_representative_frames, select_representative_frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

GROUPS = ("overall", "0_200m", "200_400m")


def _write_eval_dataset_yaml(frame_order, output_path: Path) -> Path:
    """A trivial YOLO dataset.yaml for Ultralytics' .val() (mAP only) -- train==val==every eval frame."""
    list_path = output_path.parent / "eval_images.txt"
    list_path.write_text("\n".join(str(p.resolve()) for p in frame_order) + "\n")
    output_path.write_text(
        f"train: {list_path.resolve()}\nval: {list_path.resolve()}\nnames:\n  0: vehicle\n"
    )
    return output_path


def _write_metrics_table_md(metrics: dict, standard_metrics: dict) -> str:
    rows = [
        ("Detection Rate", "detection_rate", "{:.3f}"),
        ("Precision", "precision", "{:.3f}"),
        ("False Alarms/min", "false_alarms_per_min", "{:.2f}"),
        ("Time to First Detection (s, mean)", "time_to_first_detection_mean", "{:.3f}"),
    ]
    lines = ["| Metric | Overall | 0-200 m | 200-400 m |", "|---|---|---|---|"]
    for label, key, fmt in rows:
        cells = []
        for g in GROUPS:
            v = metrics[g].get(key)
            cells.append(fmt.format(v) if isinstance(v, (int, float)) else "n/a")
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")
    map50 = standard_metrics.get("map50")
    map50_95 = standard_metrics.get("map50_95")
    lines.append(f"mAP@0.5: {map50:.3f}" if isinstance(map50, (int, float)) else "mAP@0.5: n/a")
    lines.append(f"mAP@0.5:0.95: {map50_95:.3f}" if isinstance(map50_95, (int, float)) else "mAP@0.5:0.95: n/a")
    return "\n".join(lines) + "\n"


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate_final")
def main(cfg: DictConfig) -> None:
    eval_gt_xml = Path(hydra.utils.to_absolute_path(cfg.eval_gt_xml))
    eval_images_dir = Path(hydra.utils.to_absolute_path(cfg.eval_images_dir))
    eval_labels_dir = Path(hydra.utils.to_absolute_path(cfg.eval_labels_dir))
    output_dir = Path(hydra.utils.to_absolute_path(cfg.output_dir))
    checkpoint_path = Path(cfg.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(hydra.utils.to_absolute_path(cfg.checkpoint))
    checkpoint_path = checkpoint_path.resolve()

    logger.info("Final eval checkpoint: %s", checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks, meta = parse_cvat_tracks(eval_gt_xml)
    frame_order = validate_frame_mapping(eval_images_dir, meta.n_frames)
    frame_stems = [p.stem for p in frame_order]
    logger.info("Validated %d eval frames against CVAT XML (frames 0..%d)", len(frame_order), meta.n_frames - 1)

    with Image.open(frame_order[0]) as im:
        image_w, image_h = im.size
    tracks_to_yolo_labels(tracks, meta.n_frames, image_w, image_h, eval_labels_dir, frame_stems)

    dataset_yaml = _write_eval_dataset_yaml(frame_order, output_dir / "dataset.yaml")

    if cfg.wandb.enabled:
        ensure_ultralytics_wandb_enabled()

    eval_detector = hydra.utils.instantiate(cfg.model, checkpoint=str(checkpoint_path))

    val_results = eval_detector.val(data=str(dataset_yaml), project=str(output_dir), name="val_only", exist_ok=True)
    standard_metrics = extract_standard_metrics(val_results)
    logger.info("Standard metrics: %s", standard_metrics)
    import gc

    import torch

    if getattr(eval_detector.model, "validator", None) is not None:
        del eval_detector.model.validator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(
            "GPU memory after releasing .val() state: allocated=%.2f GiB reserved=%.2f GiB",
            torch.cuda.memory_allocated() / 1024**3, torch.cuda.memory_reserved() / 1024**3,
        )

    logger.info(
        "Running inference on %d frames with FIXED final config: conf=%.3f iou=%.3f max_det=%d strategy=%s",
        len(frame_order), cfg.inference.confidence_threshold, cfg.inference.nms_iou_threshold,
        cfg.inference.max_det, cfg.inference.strategy,
    )
    predictions = run_inference(
        eval_detector,
        frame_order,
        conf=cfg.inference.confidence_threshold,
        iou=cfg.inference.nms_iou_threshold,
        strategy=cfg.inference.strategy,
        max_det=cfg.inference.max_det,
    )

    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for image_path in frame_order:
        frame_pred = predictions[str(image_path)]
        lines = predictions_to_yolo_lines(frame_pred.boxes, frame_pred.image_width, frame_pred.image_height)
        (predictions_dir / f"{image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    distance_cfg = DistanceConfig(cfg.metrics.fov_deg, cfg.metrics.car_length_m, cfg.metrics.car_width_m)
    metrics, frame_matches, ttfd_rows = run_final_evaluation(
        frame_order=frame_order,
        tracks=tracks,
        predictions=predictions,
        matching_iou_threshold=cfg.inference.matching_iou_threshold,
        distance_cfg=distance_cfg,
        band_1_max_m=cfg.metrics.band_1_max_m,
        band_2_max_m=cfg.metrics.band_2_max_m,
        fps=cfg.fps,
        exclude_frame0_tracks_from_ttfd=cfg.exclude_frame0_tracks_from_ttfd,
    )
    metrics["standard_metrics"] = standard_metrics

    write_json(str(output_dir / "metrics.json"), metrics)
    write_json(str(output_dir / "frame_matches.json"), frame_matches)
    write_ttfd_csv(ttfd_rows, output_dir / "ttfd_tracks.csv")
    (output_dir / "final_metrics_table.md").write_text(_write_metrics_table_md(metrics, standard_metrics))
    logger.info("Metrics/predictions written -> %s", output_dir)

    selection = select_representative_frames(frame_matches, ttfd_rows)
    save_representative_frames(frame_order, frame_matches, selection, output_dir / "visualizations")
    render_eval_video(frame_order, frame_matches, output_dir / "eval_predictions.mp4", fps=cfg.fps)

    wandb_run = None
    if cfg.wandb.enabled:
        entity = resolve_entity(cfg.wandb.entity)
        try:
            wandb_run = safe_wandb_init(
                project=cfg.wandb.project,
                entity=entity,
                mode=cfg.wandb.mode,
                name="vehicle-detector-final-eval",
                config={
                    "checkpoint": str(checkpoint_path),
                    "confidence_threshold": cfg.inference.confidence_threshold,
                    "nms_iou_threshold": cfg.inference.nms_iou_threshold,
                    "matching_iou_threshold": cfg.inference.matching_iou_threshold,
                    "max_det": cfg.inference.max_det,
                    "strategy": cfg.inference.strategy,
                    "fps": cfg.fps,
                    "n_frames": len(frame_order),
                    "n_gt_tracks": len(tracks),
                    "exclude_frame0_tracks_from_ttfd": cfg.exclude_frame0_tracks_from_ttfd,
                },
            )
            if wandb_run is not None:
                import wandb

                flat = {}
                for g in GROUPS:
                    for key in (
                        "tp", "fp", "fn", "detection_rate", "precision", "false_alarms_per_min",
                        "time_to_first_detection_mean", "time_to_first_detection_median",
                        "ttfd_eligible_tracks", "ttfd_detected_tracks", "ttfd_never_detected_tracks",
                    ):
                        value = metrics[g].get(key)
                        if value is not None:
                            flat[f"task/{g}/{key}"] = value
                for key, value in standard_metrics.items():
                    if value is not None:
                        flat[f"standard_metrics/{key}"] = value
                flat["n_gt_tracks"] = len(tracks)
                wandb_run.log(flat)

                table = wandb.Table(columns=["track_id", "first_visible_frame", "distance_band", "detected", "ttfd_seconds", "excluded_from_ttfd"])
                for row in sorted(ttfd_rows, key=lambda r: r["track_id"]):
                    table.add_data(row["track_id"], row["first_visible_frame"], row["distance_band"], row["detected"], row["ttfd_seconds"], row["excluded_from_ttfd"])
                wandb_run.log({"ttfd_tracks": table})

                for image_path in sorted((output_dir / "visualizations").glob("*.jpg")):
                    wandb_run.log({f"visualizations/{image_path.stem}": wandb.Image(str(image_path))})
        finally:
            safe_wandb_finish(wandb_run)

    logger.info("Final evaluation complete -> %s", output_dir)


if __name__ == "__main__":
    main()
