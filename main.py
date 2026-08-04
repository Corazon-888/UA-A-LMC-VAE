import sys

import hydra
import lightning as L
from fontTools.misc.plistlib import totree
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf, ListConfig
import time

import torch
import wandb

from lightning_modules.lightning_cm import LightningConsistencyModel
from utils.callback_utils import get_callbacks, get_delete_checkpoints_callback
from utils.datamodule_utils import get_datamodule
from utils.naming_utils import get_run_name
from utils.model_utils import get_model
from utils.training_steps import get_model_step, get_trainer_max_steps
from wandb_config import key
from lightning.pytorch.utilities import rank_zero_only
from pathlib import Path


RUNTIME_OVERRIDE_KEYS = (
    'log_frequency',
    'compute_fid',
    'compute_rec_fid',
    'log_samples',
    'log_rec',
    'enable_progress_bar',
)


def get_explicit_runtime_overrides(cfg: DictConfig) -> dict:
    task_overrides = HydraConfig.get().overrides.task
    explicit_keys = {
        override.lstrip('+~').split('=', 1)[0]
        for override in task_overrides
        if '=' in override
    }
    return {
        key: OmegaConf.select(cfg, key)
        for key in RUNTIME_OVERRIDE_KEYS
        if key in explicit_keys
    }


def restore_runtime_config(checkpoint_cfg: DictConfig, runtime_cfg: DictConfig,
                           explicit_overrides: dict) -> DictConfig:
    # Operational settings may change when resuming without changing the experiment itself.
    checkpoint_cfg.root_dir = runtime_cfg.root_dir
    checkpoint_cfg.dataset.data_dir = runtime_cfg.dataset.data_dir
    checkpoint_cfg.log_model = runtime_cfg.log_model
    for key, value in explicit_overrides.items():
        OmegaConf.update(checkpoint_cfg, key, value, merge=False)
    return checkpoint_cfg


def restore_model_step(model: LightningConsistencyModel, checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    model.step = get_model_step(
        ckpt['global_step'],
        model.cfg.model.gan_warmup_steps,
        model.cfg.model.use_gan,
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_cfg = cfg
    explicit_runtime_overrides = get_explicit_runtime_overrides(cfg)
    checkpoint_path = cfg.get('ckpt_path', '')
    if cfg.reload and checkpoint_path:
        raise ValueError('Use either reload=True or ckpt_path, not both.')

    if checkpoint_path:
        reload = True
        run_id = Path(cfg.run_path).name if cfg.run_path else None
        resume = 'must' if run_id else 'allow'
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
        model = LightningConsistencyModel.load_from_checkpoint(checkpoint_path, weights_only=False)
        restore_model_step(model, checkpoint_path)
        cfg = restore_runtime_config(model.cfg, runtime_cfg, explicit_runtime_overrides)
        L.seed_everything(cfg.seed, workers=True)
    elif cfg.reload:
        reload = True
        #checkpoint_path = f'{cfg.root_dir}/model.ckpt'
        wandb.login(key=key)
        run_path = Path(cfg.run_path)
        run_id = run_path.name
        resume = 'must'
        run_path = run_path.with_name(f'model-{run_path.name}')
        checkpoint_reference = f'{run_path}:latest'
        checkpoint_path = Path(cfg.root_dir) / "model.ckpt"
        logger = WandbLogger(resume='must', id=run_id)
        logger.download_artifact(checkpoint_reference, save_dir=cfg.root_dir, artifact_type="model")
        while True:
            # ugly hack to make sure model reloading works with multi gpu
            try:
                model = LightningConsistencyModel.load_from_checkpoint(checkpoint_path, weights_only=False)
                break
            except:
                time.sleep(30)
        restore_model_step(model, checkpoint_path)
        cfg = restore_runtime_config(model.cfg, runtime_cfg, explicit_runtime_overrides)
        # cfg.log_frequency = 2500
        L.seed_everything(cfg.seed, workers=True)
    else:
        L.seed_everything(cfg.seed, workers=True)
        reload = False
        resume = 'allow'
        run_id = None
        model = get_model(cfg)
        model = LightningConsistencyModel(cfg, model)

    if cfg.devices == 'auto':
        num_of_gpus = torch.cuda.device_count()
    elif isinstance(cfg.devices, list) or isinstance(cfg.devices, ListConfig):
        num_of_gpus = len(cfg.devices)
    else:
        num_of_gpus = 1

    if num_of_gpus > 1:
        cfg['batch_multiplier'] = num_of_gpus

    name = get_run_name(cfg)
    dm = get_datamodule(cfg)
    callbacks = get_callbacks(cfg)

    if cfg.use_logger:
        wandb.login(key=key)
        # depending on the case, set log_model=True to log only at the end, log_model="all" to log during training (in case training might be interrupted)
        logger = WandbLogger(project=cfg.project, name=name, log_model=cfg.log_model, save_dir=cfg.root_dir, resume=resume, id=run_id)

        config_dictionary = dict(
            cfg
        )
        if rank_zero_only.rank == 0:
            logger.experiment.config.update(config_dictionary, allow_val_change=True)
            if cfg.log_model:
                callbacks.append(get_delete_checkpoints_callback(cfg, logger.experiment.path))
    else:
        logger = False

    total_training_steps = get_trainer_max_steps(
        cfg.model.total_training_steps,
        cfg.model.gan_warmup_steps,
        cfg.model.use_gan,
    )

    trainer = L.Trainer(max_steps=total_training_steps,
                        logger=logger,
                        strategy=cfg.strategy,
                        devices=cfg.devices,
                        callbacks=callbacks,
                        log_every_n_steps=cfg.log_frequency,
                        precision=cfg.precision,
                        accumulate_grad_batches=cfg.accumulate_grad_batches,
                        fast_dev_run=cfg.fast_dev_run,
                        enable_progress_bar=cfg.enable_progress_bar,
                        accelerator=cfg.accelerator,
                        default_root_dir=cfg.root_dir,
                        deterministic=cfg.deterministic,
                        sync_batchnorm=cfg.sync_batchnorm,
                        )
    if reload:
        dm.prepare_data()
        trainer.fit(model=model, datamodule=dm, ckpt_path=checkpoint_path, weights_only=False)
    else:
        trainer.fit(model=model, datamodule=dm)
    time.sleep(10)
    if rank_zero_only.rank == 0 and cfg.use_logger and cfg.log_model:
        for artifact_version in wandb.Api().run(logger.experiment.path).logged_artifacts():
            # Keep only artifacts with alias "best" or "latest"
            if len(artifact_version.aliases) == 0:
                artifact_version.delete()

    sys.exit(0)


if __name__ == "__main__":
    main()
