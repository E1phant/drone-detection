"""Leave-one-video-out detector training + task-specific evaluation.

Usage:
    poetry run python scripts/train_detector.py fold=A   # or B, C, D
    poetry run python scripts/train_detector.py fold=all  # final A+B+C+D model, no held-out eval
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.lovo_split import build_fold, discover_frames, ensure_labels_symlink, resolve_video_id
from src.metrics.confidence_sweep import log_confidence_sweep_to_wandb, run_confidence_sweep, write_confidence_sweep_csv
from src.metrics.task_evaluator import DistanceConfig, evaluate_fold, flatten_task_metrics_for_wandb
from src.models.inference import filter_predictions_by_confidence, predictions_to_yolo_lines, run_inference
from src.models.yolo_detector import YoloDetector, extract_standard_metrics
from src.utils.io import write_json
from src.utils.wandb_helpers import (
    ensure_ultralytics_wandb_enabled,
    resolve_entity,
    safe_wandb_finish,
    safe_wandb_init,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _wandb_run_name(exp_name: str) -> str:
    return f"vehicle-detector-fold-{exp_name}" if exp_name != "all" else "vehicle-detector-final"


def _wandb_config(cfg: DictConfig, held_out_video, train_videos) -> dict:
    return {
        "fold": cfg.fold,
        "held_out_video": held_out_video,
        "train_videos": train_videos,
        "model_checkpoint": cfg.model.checkpoint,
        "image_size": cfg.model.image_size,
        "batch_size": cfg.trainer.batch_size,
        "epochs": cfg.trainer.epochs,
        "learning_rate": cfg.trainer.learning_rate,
        "optimizer": cfg.trainer.optimizer,
        "confidence_threshold": cfg.inference.confidence_threshold,
        "nms_iou_threshold": cfg.inference.nms_iou_threshold,
        "matching_iou_threshold": cfg.inference.matching_iou_threshold,
        "seed": cfg.trainer.seed,
        "confidence_sweep_enabled": cfg.confidence_sweep.enabled,
        "confidence_sweep_thresholds": list(cfg.confidence_sweep.thresholds),
    }


def _make_periodic_sweep_callback(cfg: DictConfig, val_records, distance_cfg: DistanceConfig, wandb_run):
    """Every `every_n_epochs`, reload the just-saved last.pt and log a confidence sweep at that epoch's step."""

    def _callback(trainer):
        epoch = trainer.epoch + 1
        if epoch % cfg.confidence_sweep.every_n_epochs != 0:
            return
        if wandb_run is None:
            return

        ckpt_detector = None
        try:
            ckpt_detector = YoloDetector.from_checkpoint(str(trainer.last), image_size=cfg.model.image_size)
            base_predictions = run_inference(
                ckpt_detector,
                [r.image_path for r in val_records],
                conf=min(cfg.confidence_sweep.thresholds),
                iou=cfg.inference.nms_iou_threshold,
                strategy=cfg.inference.strategy,
                max_det=cfg.inference.max_det,
            )
            sweep = run_confidence_sweep(
                fold=cfg.fold,
                checkpoint=str(trainer.last),
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
            log_confidence_sweep_to_wandb(wandb_run, sweep, step=epoch, key_prefix="confidence_sweep_epoch")
            logger.info("Periodic confidence sweep logged at epoch %d", epoch)
        except Exception:
            logger.warning("Periodic confidence sweep at epoch %d failed -- continuing training", epoch, exc_info=True)
        finally:
            if ckpt_detector is not None:
                del ckpt_detector
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass

    return _callback


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    fold = cfg.fold
    exp_name = cfg.exp_name

    train_dataset_dir = Path(hydra.utils.to_absolute_path(cfg.train_dataset_dir))
    images_dir = Path(hydra.utils.to_absolute_path(cfg.images_dir))
    gt_labels_dir = Path(hydra.utils.to_absolute_path(cfg.gt_labels_dir))
    splits_dir = Path(hydra.utils.to_absolute_path(cfg.splits_dir))
    outputs_dir = Path(hydra.utils.to_absolute_path(cfg.outputs_dir))

    ensure_labels_symlink(train_dataset_dir)
    frames = discover_frames(images_dir, gt_labels_dir)
    known_videos = sorted({r.video_id for r in frames})

    video_id = resolve_video_id(fold, known_videos) if fold != "all" else "all"
    held_out_video = video_id if fold != "all" else None
    train_videos = sorted(v for v in known_videos if v != held_out_video) if held_out_video else known_videos
    build_fold(frames, video_id, splits_dir, OmegaConf.to_container(cfg.class_names), fold_label=fold)
    dataset_yaml = splits_dir / f"fold_{fold}" / "dataset.yaml"

    val_records = (
        sorted((r for r in frames if r.video_id == video_id), key=lambda r: r.image_path)
        if fold != "all"
        else []
    )
    distance_cfg = DistanceConfig(cfg.metrics.fov_deg, cfg.metrics.car_length_m, cfg.metrics.car_width_m)

    entity = resolve_entity(cfg.wandb.entity) if cfg.wandb.enabled else None
    wandb_run = None
    wandb_run_id = None
    if cfg.wandb.enabled:
        wandb_run = safe_wandb_init(
            project=cfg.wandb.project,
            entity=entity,
            mode=cfg.wandb.mode,
            name=_wandb_run_name(exp_name),
            config=_wandb_config(cfg, held_out_video, train_videos),
        )
        wandb_run_id = wandb_run.id if wandb_run is not None else None

    if cfg.wandb.enabled:
        ensure_ultralytics_wandb_enabled()

    detector = hydra.utils.instantiate(cfg.model)
    if fold != "all" and cfg.confidence_sweep.enabled and cfg.confidence_sweep.every_n_epochs > 0:
        detector.model.add_callback(
            "on_fit_epoch_end", _make_periodic_sweep_callback(cfg, val_records, distance_cfg, wandb_run)
        )

    detector.train(
        data=str(dataset_yaml),
        epochs=cfg.trainer.epochs,
        batch=cfg.trainer.batch_size,
        lr0=cfg.trainer.learning_rate,
        optimizer=cfg.trainer.optimizer,
        device=cfg.trainer.device,
        workers=cfg.trainer.workers,
        seed=cfg.trainer.seed,
        patience=cfg.trainer.patience,
        deterministic=True,
        project=str(outputs_dir),  # must be absolute: Ultralytics rebases a
        name=f"fold_{fold}",       # relative project under its own SETTINGS['runs_dir']
        exist_ok=True,
    )
    fold_dir = detector.save_dir
    best_checkpoint = detector.best_checkpoint_path
    logger.info("Resolved best checkpoint from Ultralytics trainer: %s", best_checkpoint)
    if not best_checkpoint.exists():
        raise FileNotFoundError(
            f"Ultralytics reported the best checkpoint at {best_checkpoint} but that file does not exist."
        )
    standard_metrics = extract_standard_metrics(getattr(detector.model, "metrics", None))

    if wandb_run_id:
        (fold_dir / "wandb_run_id.txt").write_text(wandb_run_id + "\n")

    if fold == "all":
        logger.warning(
            "fold=all: no genuinely held-out video exists for the final A+B+C+D model. "
            "The task-specific evaluator (including the confidence sweep) is skipped -- "
            "this model is evaluated later, only against the official held-out eval video."
        )
        write_json(
            str(fold_dir / "metrics.json"),
            {
                "fold": "all",
                "held_out_video": None,
                "task_evaluation_skipped_reason": (
                    "fold=all trains on every video combined; there is no held-out video to "
                    "evaluate against here. Evaluate this checkpoint later against the official eval video."
                ),
                "standard_metrics": standard_metrics,
            },
        )
        try:
            if wandb_run_id:
                wandb_run = safe_wandb_init(
                    project=cfg.wandb.project, entity=entity, mode=cfg.wandb.mode, id=wandb_run_id, resume="must"
                )
            if wandb_run is not None:
                wandb_run.log(flatten_task_metrics_for_wandb({}, standard_metrics))
        finally:
            safe_wandb_finish(wandb_run)
        return

    try:
        eval_detector = hydra.utils.instantiate(cfg.model, checkpoint=str(best_checkpoint))
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
            held_out_video=held_out_video,
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
        logger.info("Fold %s complete -> %s", fold, fold_dir / "metrics.json")

        sweep = None
        if sweep_enabled:
            sweep = run_confidence_sweep(
                fold=fold,
                checkpoint=str(best_checkpoint),
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

        if wandb_run_id:
            wandb_run = safe_wandb_init(
                project=cfg.wandb.project, entity=entity, mode=cfg.wandb.mode, id=wandb_run_id, resume="must"
            )
        if wandb_run is not None:
            wandb_run.log(flatten_task_metrics_for_wandb(metrics, standard_metrics))
            if sweep is not None:
                log_confidence_sweep_to_wandb(wandb_run, sweep)  # no step -- final/summary sweep for this run
    finally:
        safe_wandb_finish(wandb_run)


if __name__ == "__main__":
    main()
