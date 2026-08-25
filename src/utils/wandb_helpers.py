import logging
from typing import Optional

import wandb

logger = logging.getLogger(__name__)


def ensure_ultralytics_wandb_enabled() -> None:
    from ultralytics.utils import SETTINGS

    if SETTINGS.get("wandb") is not True:
        logger.info("Ultralytics SETTINGS['wandb'] was disabled -- enabling it so training metrics reach W&B")
        SETTINGS.update({"wandb": True})


def resolve_entity(cfg_entity: Optional[str]) -> Optional[str]:
    if cfg_entity:
        return cfg_entity
    try:
        return wandb.Api().default_entity
    except Exception:
        logger.warning("Could not resolve a default W&B entity", exc_info=True)
        return None


def find_existing_run_id(project: str, entity: Optional[str], run_name: str) -> Optional[str]:
    if not entity:
        return None
    try:
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
        runs = sorted(runs, key=lambda r: r.created_at, reverse=True)
        return runs[0].id if runs else None
    except Exception:
        logger.warning("W&B run lookup for %r failed", run_name, exc_info=True)
        return None


def safe_wandb_init(**kwargs):
    try:
        return wandb.init(**kwargs)
    except Exception:
        logger.warning("W&B init failed -- continuing without W&B tracking", exc_info=True)
        return None


def safe_wandb_finish(run) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        logger.warning("W&B finish failed", exc_info=True)
