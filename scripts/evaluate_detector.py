import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.lovo_split import build_fold, discover_frames, ensure_labels_symlink, resolve_video_id
from src.metrics.confidence_sweep import log_confidence_sweep_to_wandb, run_confidence_sweep, write_confidence_sweep_csv
from src.metrics.task_evaluator import DistanceConfig, evaluate_fold, flatten_task_metrics_for_wandb
from src.models.inference import filter_predictions_by_confidence, predictions_to_yolo_lines, run_inference
from src.models.yolo_detector import extract_standard_metrics
from src.utils.io import write_json
from src.utils.wandb_helpers import (
    ensure_ultralytics_wandb_enabled,
    find_existing_run_id,
    resolve_entity,
    safe_wandb_finish,
    safe_wandb_init,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _resolve_fold_dir(checkpoint_path: Path) -> Path:
    """Given .../<save_dir>/weights/best.pt, return <save_dir>; else its parent, with a warning."""
    if checkpoint_path.parent.name == "weights":
        return checkpoint_path.parent.parent
    logger.warning(
        "Checkpoint %s is not at the usual '<save_dir>/weights/<file>.pt' location -- "
        "writing predictions/metrics.json next to it instead, in %s",
        checkpoint_path, checkpoint_path.parent,
    )
    return checkpoint_path.parent


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    fold = cfg.fold

    train_dataset_dir = Path(hydra.utils.to_absolute_path(cfg.train_dataset_dir))
    images_dir = Path(hydra.utils.to_absolute_path(cfg.images_dir))
    gt_labels_dir = Path(hydra.utils.to_absolute_path(cfg.gt_labels_dir))
    splits_dir = Path(hydra.utils.to_absolute_path(cfg.splits_dir))

    checkpoint_path = Path(cfg.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(hydra.utils.to_absolute_path(cfg.checkpoint))
    checkpoint_path = checkpoint_path.resolve()
    logger.info("Evaluating checkpoint: %s", checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    fold_dir = _resolve_fold_dir(checkpoint_path)
    fold_dir.mkdir(parents=True, exist_ok=True)

    ensure_labels_symlink(train_dataset_dir)
    frames = discover_frames(images_dir, gt_labels_dir)
    known_videos = sorted({r.video_id for r in frames})
    video_id = resolve_video_id(fold, known_videos)

    build_fold(frames, video_id, splits_dir, OmegaConf.to_container(cfg.class_names), fold_label=fold)
    dataset_yaml = splits_dir / f"fold_{fold}" / "dataset.yaml"
    val_records = sorted((r for r in frames if r.video_id == video_id), key=lambda r: r.image_path)

    if cfg.wandb.enabled:
        ensure_ultralytics_wandb_enabled()

    eval_detector = hydra.utils.instantiate(cfg.model, checkpoint=str(checkpoint_path))

    val_results = eval_detector.val(
        data=str(dataset_yaml), project=str(fold_dir), name="val_only", exist_ok=True
    )
    standard_metrics = extract_standard_metrics(val_results)

    distance_cfg = DistanceConfig(cfg.metrics.fov_deg, cfg.metrics.car_length_m, cfg.metrics.car_width_m)
    sweep_enabled = cfg.confidence_sweep.enabled
    base_conf = (
        min(cfg.inference.confidence_threshold, min(cfg.confidence_sweep.thresholds))
        if sweep_enabled
        else cfg.inference.confidence_threshold
    )
    base_predictions = run_inference(
        eval_detector,
        [r.image_path for r in val_records],
        conf=base_conf,
        iou=cfg.inference.nms_iou_threshold,
        strategy=cfg.inference.strategy,
        max_det=cfg.inference.max_det,
    )
    predictions = (
        filter_predictions_by_confidence(base_predictions, cfg.inference.confidence_threshold)
        if sweep_enabled
        else base_predictions
    )

    predictions_dir = fold_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for record in val_records:
        frame_pred = predictions[str(record.image_path)]
        lines = predictions_to_yolo_lines(frame_pred.boxes, frame_pred.image_width, frame_pred.image_height)
        (predictions_dir / f"{record.image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    metrics, detection_records = evaluate_fold(
        fold=fold,
        held_out_video=video_id,
        val_records=val_records,
        predictions=predictions,
        matching_iou_threshold=cfg.inference.matching_iou_threshold,
        distance_cfg=distance_cfg,
        band_1_max_m=cfg.metrics.band_1_max_m,
        band_2_max_m=cfg.metrics.band_2_max_m,
        fps=cfg.data.fps,
    )
    metrics["standard_metrics"] = standard_metrics

    write_json(str(predictions_dir / "task_predictions.json"), detection_records)
    write_json(str(fold_dir / "metrics.json"), metrics)
    logger.info("Evaluation complete for fold %s -> %s", fold, fold_dir / "metrics.json")

    sweep = None
    if sweep_enabled:
        sweep = run_confidence_sweep(
            fold=fold,
            checkpoint=str(checkpoint_path),
            val_records=val_records,
            base_predictions=base_predictions,
            thresholds=cfg.confidence_sweep.thresholds,
            matching_iou_threshold=cfg.inference.matching_iou_threshold,
            distance_cfg=distance_cfg,
            band_1_max_m=cfg.metrics.band_1_max_m,
            band_2_max_m=cfg.metrics.band_2_max_m,
            fps=cfg.data.fps,
            max_det=cfg.inference.max_det,
        )
        write_json(str(fold_dir / "confidence_sweep.json"), sweep)
        write_confidence_sweep_csv(sweep, fold_dir / "confidence_sweep.csv")
        logger.info("Confidence sweep complete -> %s", fold_dir / "confidence_sweep.json")

    wandb_run = None
    if cfg.wandb.enabled:
        entity = resolve_entity(cfg.wandb.entity)
        run_name = f"vehicle-detector-fold-{fold}"
        run_id = None

        run_id_file = fold_dir / "wandb_run_id.txt"
        if run_id_file.exists():
            run_id = run_id_file.read_text().strip()
            logger.info("Found recorded W&B run id for this fold: %s", run_id)
        else:
            run_id = find_existing_run_id(cfg.wandb.project, entity, run_name)
            if run_id:
                logger.info("Found existing W&B run %r by name lookup: %s", run_name, run_id)

        try:
            if run_id:
                wandb_run = safe_wandb_init(
                    project=cfg.wandb.project, entity=entity, mode=cfg.wandb.mode, id=run_id, resume="must"
                )
            if wandb_run is None:
                logger.warning(
                    "No existing W&B run found for fold %s -- creating a new '%s-eval' run "
                    "instead of the original training run.", fold, run_name
                )
                wandb_run = safe_wandb_init(
                    project=cfg.wandb.project, entity=entity, mode=cfg.wandb.mode, name=f"{run_name}-eval",
                    config={"fold": fold, "held_out_video": video_id, "checkpoint": str(checkpoint_path)},
                )
            if wandb_run is not None:
                wandb_run.log(flatten_task_metrics_for_wandb(metrics, standard_metrics))
                if sweep is not None:
                    log_confidence_sweep_to_wandb(wandb_run, sweep)  # no step -- final/summary sweep for this run
        finally:
            safe_wandb_finish(wandb_run)


if __name__ == "__main__":
    main()
