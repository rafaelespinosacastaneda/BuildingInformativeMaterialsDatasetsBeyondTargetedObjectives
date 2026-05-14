#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simplified ALIGNN sklearn interface for materials science machine learning.

Uses the nfflr package for ALIGNN model and training utilities.
"""

import gc
import json
import logging
import os
import pathlib
import warnings
from dataclasses import dataclass
from typing import Union, Tuple, List, Optional

import dgl
import ignite
import nfflr
import pandas as pd
import torch
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Events, create_supervised_evaluator
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, TerminateOnNan
from ignite.metrics import Loss, MeanAbsoluteError, RootMeanSquaredError
from jarvis.core.atoms import Atoms as JAtoms
from jarvis.io.vasp.inputs import Poscar
from nfflr.models import ALIGNN, ALIGNNConfig
from nfflr.nn import PeriodicRadiusGraph
from nfflr.train.utils import group_decay, setup_optimizer
from pymatgen.core.structure import Structure
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)
tqdm.pandas()

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PLACEHOLDER_TARGET_VALUE = -9999


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class NfflrAlignnConfig:
    """Configuration for nfflr ALIGNN model and training.

    Holds both model architecture parameters (passed to ALIGNNConfig) and
    training hyperparameters (used by the ignite training loop).
    """

    # --- graph transform ---
    cutoff: float = 8.0

    # --- model architecture ---
    alignn_layers: int = 2
    gcn_layers: int = 4
    norm: str = "batchnorm"          # "batchnorm" or "layernorm"
    atom_features: str = "cgcnn"
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 256
    output_features: int = 1

    # --- training ---
    epochs: int = 500
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    criterion: str = "mse"           # "mse", "l1", "poisson"
    scheduler: str = "onecycle"      # "onecycle", "step", or "none"
    lr_step_size: int = 10            # epochs between LR decay steps (for "step" scheduler)
    warmup_steps: float = 0.3        # fraction of training for LR warmup
    random_seed: int = 42

    # --- data / logging ---
    target: str = "target"
    id_tag: str = "jid"
    output_dir: str = "output_alignn/"
    num_workers: int = 0
    pin_memory: bool = False
    write_checkpoint: bool = True
    progress: bool = True
    classification_threshold: Optional[float] = None

    # --- gradient clipping ---
    max_grad_norm: float = 1.0       # clips gradient L2-norm; 0.0 = disabled

    # --- mixed precision ---
    use_amp: bool = True             # float16 activations; ~halves peak per-batch memory

    # --- early stopping ---
    early_stopping_patience: int = 20     # 0 = disabled
    early_stopping_min_delta: float = 1e-4  # minimum improvement to count as progress

    @classmethod
    def from_json(cls, path: str) -> "NfflrAlignnConfig":
        """Load config from a JSON file.

        Accepts both the new flat format and the old alignn nested format
        (with a "model" sub-dict).
        """
        with open(path, encoding='utf-8') as f:
            d = json.load(f)

        # Flatten nested 'model' sub-dict if present (old alignn format)
        model_cfg = d.pop("model", {})
        model_cfg.pop("name", None)
        d.update(model_cfg)

        # Strip old alignn-only keys
        _old_keys = {
            "version", "neighbor_strategy", "use_canonize", "max_neighbors",
            "atom_input_features", "link", "zero_inflated", "classification",
            "extra_features", "standard_scalar_and_pca", "log_tensorboard",
            "save_dataloader", "write_predictions", "keep_data_order",
            "distributed", "n_early_stopping", "dataset",
            "classification_threshold", "store_outputs",
        }
        for k in _old_keys:
            d.pop(k, None)

        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_alignn_config(self) -> ALIGNNConfig:
        """Build an nfflr ALIGNNConfig from this training config."""
        transform = PeriodicRadiusGraph(cutoff=self.cutoff)
        return ALIGNNConfig(
            transform=transform,
            alignn_layers=self.alignn_layers,
            gcn_layers=self.gcn_layers,
            norm=self.norm,
            atom_features=self.atom_features,
            edge_input_features=self.edge_input_features,
            triplet_input_features=self.triplet_input_features,
            embedding_features=self.embedding_features,
            hidden_features=self.hidden_features,
            output_features=self.output_features,
        )


# =============================================================================
# Utility Functions
# =============================================================================

def structure2atoms(structure: Structure) -> JAtoms:
    """Convert pymatgen Structure to jarvis Atoms object."""
    return Poscar.from_string(structure.to(fmt="poscar")).atoms


def _to_nfflr_atoms(item) -> nfflr.Atoms:
    """Convert various atom representations to nfflr.Atoms.

    Accepts: nfflr.Atoms, jarvis Atoms, jarvis Atoms dict, pymatgen Structure.
    """
    if isinstance(item, nfflr.Atoms):
        return item
    if isinstance(item, JAtoms):
        return nfflr.Atoms(item)
    if isinstance(item, dict):
        return nfflr.Atoms(JAtoms.from_dict(item))
    if isinstance(item, Structure):
        return nfflr.Atoms(structure2atoms(item))
    raise TypeError(f"Cannot convert {type(item).__name__} to nfflr.Atoms")


# =============================================================================
# Dataset Class
# =============================================================================

class AlignnDataset(torch.utils.data.Dataset):
    """Dataset for nfflr ALIGNN: stores nfflr.Atoms and applies PeriodicRadiusGraph lazily.

    The graph transform (PeriodicRadiusGraph) is applied in __getitem__ so
    graphs are built on-the-fly during DataLoader iteration. The nfflr ALIGNN
    model internally computes the line graph from the crystal graph, so only
    a single DGLGraph per sample is returned.
    """

    def __init__(
        self,
        atoms_list: List[nfflr.Atoms],
        labels: List[float],
        transform: PeriodicRadiusGraph,
        ids: Optional[List] = None,
    ):
        self.atoms_list = atoms_list
        self.labels = torch.tensor(labels, dtype=torch.get_default_dtype())
        self.transform = transform
        self.ids = ids if ids is not None else list(range(len(atoms_list)))

    def __len__(self) -> int:
        return len(self.atoms_list)

    def __getitem__(self, idx: int) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        graph = self.transform(self.atoms_list[idx])
        return graph, self.labels[idx]

    @staticmethod
    def collate(samples: List[Tuple[dgl.DGLGraph, torch.Tensor]]):
        """Batch graphs across samples for the DataLoader."""
        graphs, targets = map(list, zip(*samples))
        return dgl.batch(graphs), torch.stack(targets)

    @staticmethod
    def prepare_batch(
        batch: Tuple[dgl.DGLGraph, torch.Tensor],
        device=None,
        non_blocking: bool = False,
    ) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        """Send batch to device."""
        g, targets = batch
        return (
            g.to(device, non_blocking=non_blocking),
            targets.to(device, non_blocking=non_blocking),
        )


# =============================================================================
# DataLoader Helpers
# =============================================================================

def _build_atoms_list(series: pd.Series) -> List[nfflr.Atoms]:
    """Convert a Series of atom representations to a list of nfflr.Atoms."""
    return [_to_nfflr_atoms(a) for a in tqdm(series, desc="Converting atoms")]


def get_loader(
    df: Union[pd.DataFrame, pd.Series],
    config: NfflrAlignnConfig,
    drop_last: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for crystal structure data.

    Args:
        df: DataFrame or Series whose first column holds atom representations
            (jarvis Atoms dict, nfflr.Atoms, or pymatgen Structure). If only
            one column is present, a placeholder target column is added.
        config: Training / model configuration
        drop_last: Drop the last incomplete batch
        shuffle: Shuffle the data

    Returns:
        DataLoader backed by AlignnDataset
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()
    else:
        df = df.copy()

    # Add placeholder target for prediction-only calls
    if df.shape[1] == 1:
        df[config.target] = PLACEHOLDER_TARGET_VALUE

    # Ensure ID column
    if config.id_tag not in df.columns:
        df[config.id_tag] = df.index.tolist()

    atoms_col = df.iloc[:, 0]
    atoms_list = _build_atoms_list(atoms_col)
    labels = df[config.target].to_numpy()
    ids = df.index.tolist()

    transform = PeriodicRadiusGraph(cutoff=config.cutoff)
    dataset = AlignnDataset(
        atoms_list=atoms_list,
        labels=labels,
        transform=transform,
        ids=ids,
    )

    # Never drop the only batch: if dataset is smaller than batch_size,
    # dropping the last batch would leave zero batches and crash OneCycleLR.
    effective_drop_last = drop_last and (len(dataset) >= config.batch_size)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=AlignnDataset.collate,
        drop_last=effective_drop_last,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )


def _create_data_loaders(
    X: Union[pd.DataFrame, pd.Series],
    y: Union[pd.DataFrame, pd.Series],
    config: NfflrAlignnConfig,
    val_data: Optional[Tuple] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create train and validation DataLoaders.

    Args:
        X: Training features (atom representations)
        y: Training targets
        config: Training configuration
        val_data: Optional (X_val, y_val) tuple

    Returns:
        (train_loader, val_loader)
    """

    def _make_df(features, targets):
        if isinstance(features, pd.Series):
            df = pd.concat([features, targets], axis=1)
            df.columns = [features.name or "atoms", config.target]
        else:
            df = pd.concat([features, targets], axis=1)
            df.columns = [*features.columns, config.target]
        return df

    train_loader = get_loader(_make_df(X, y), config, drop_last=True, shuffle=True)

    val_loader = None
    if val_data is not None:
        val_loader = get_loader(
            _make_df(val_data[0], val_data[1]), config, drop_last=False, shuffle=False
        )

    return train_loader, val_loader


# =============================================================================
# Model Initialization and Training
# =============================================================================

def _init_model(
    model_instance,
    config: NfflrAlignnConfig,
    chk_file: Optional[str] = None,
    reset_parameters: bool = True,
) -> None:
    """Attach config/device to model and optionally load a checkpoint."""
    model_instance.training_config = config
    model_instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pathlib.Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    if chk_file is not None:
        model_instance.load_state_dict(
            torch.load(chk_file, map_location=model_instance.device)["model"]
        )
        logger.info(f"Checkpoint {chk_file} loaded")
        model_instance.reset_parameters = False
    else:
        model_instance.reset_parameters = reset_parameters
        torch.save(
            model_instance.state_dict(),
            os.path.join(config.output_dir, "model_initial.pth"),
        )

    model_instance.to(model_instance.device)


def _fit_model(
    model_instance,
    X: Union[pd.DataFrame, pd.Series],
    y: Union[pd.DataFrame, pd.Series],
    val: Optional[Tuple] = None,
) -> None:
    """Train the ALIGNN model."""
    config = model_instance.training_config

    # Move model to GPU for training (it may have been offloaded to CPU).
    model_instance.to(model_instance.device)

    if model_instance.reset_parameters:
        initial_path = os.path.join(config.output_dir, "model_initial.pth")
        model_instance.load_state_dict(torch.load(initial_path))

    train_loader, val_loader = _create_data_loaders(X, y, config, val)
    trainer = _get_trainer(model_instance, train_loader, val_loader)
    try:
        trainer.run(train_loader, max_epochs=config.epochs)
    finally:
        # Always release GPU memory, even if training crashes with OOM.
        # Clear ignite engine state so it releases any retained batch tensors.
        if hasattr(trainer, "state") and trainer.state is not None:
            trainer.state.output = None
            trainer.state.batch = None
        del train_loader, val_loader, trainer
        # Move model weights to CPU so they don't occupy GPU VRAM between fits.
        model_instance.cpu()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()


def _predict_with_model(
    model_instance, X: Union[pd.DataFrame, pd.Series]
) -> pd.Series:
    """Run inference and return predictions as a Series indexed by sample ID."""
    # Move model back to GPU for inference.
    model_instance.to(model_instance.device)
    model_instance.eval()
    config = model_instance.training_config

    test_loader = get_loader(X, config, drop_last=False, shuffle=False)
    col_ids = test_loader.dataset.ids
    col_pred: List[float] = []

    try:
        with torch.no_grad():
            for g, _targets in test_loader:
                g = g.to(model_instance.device)
                out = model_instance(g)
                out = out.cpu().numpy()
                if out.ndim == 0:
                    col_pred.append(float(out))
                elif out.ndim == 1:
                    col_pred.extend(out.tolist())
                else:
                    col_pred.extend(out.squeeze().tolist())
    finally:
        del test_loader
        # Move model back to CPU after inference to free VRAM.
        model_instance.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return pd.Series(data=col_pred, index=col_ids, name=config.target)


def _setup_scheduler(config: NfflrAlignnConfig, optimizer, steps_per_epoch: int):
    """Build a learning-rate scheduler from config."""
    if config.scheduler == "none" or config.scheduler is None:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)

    if config.scheduler == "onecycle":
        warmup = config.warmup_steps
        pct_start = warmup if warmup < 1 else warmup / (config.epochs * steps_per_epoch)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
        )

    if config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size)

    raise ValueError(f"Unknown scheduler: {config.scheduler!r}")


def _get_trainer(model_instance, train_loader, val_loader):
    """Build and return an ignite training engine.

    Uses nfflr utilities (group_decay, setup_optimizer) and the standard
    ignite supervised trainer / evaluator pattern.
    """
    config = model_instance.training_config

    if val_loader is None:
        val_loader = train_loader

    # --- optimizer / scheduler ---
    params = group_decay(model_instance)
    optimizer = setup_optimizer(params, config)
    scheduler = _setup_scheduler(config, optimizer, len(train_loader))

    # --- loss function ---
    criteria = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "poisson": nn.PoissonNLLLoss(log_input=False, full=True),
    }
    criterion = criteria[config.criterion]

    # --- metrics ---
    metrics = {
        "loss": Loss(criterion),
        "mae": MeanAbsoluteError(),
        "rmse": RootMeanSquaredError(),
    }

    # --- seeding ---
    deterministic = False
    if config.random_seed is not None:
        deterministic = True
        ignite.utils.manual_seed(config.random_seed)

    prepare_batch = AlignnDataset.prepare_batch
    device = model_instance.device

    # AMP scaler: only active when use_amp=True and a CUDA device is available.
    use_amp = config.use_amp and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        logger.info("AMP enabled: using float16 activations to reduce peak GPU memory.")

    # OOM-safe training step: skip batches that exceed GPU memory rather than
    # crashing the entire epoch.  With AMP enabled, peak activation memory is
    # roughly halved so OOM should not occur in normal operation.
    _oom_skipped = [0]

    def _oom_safe_process_fn(engine, batch):
        model_instance.train()
        optimizer.zero_grad(set_to_none=True)
        try:
            g, targets = prepare_batch(batch, device=device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model_instance(g)
                loss = criterion(outputs, targets)
            loss_val = loss.item()
            if not torch.isfinite(loss):
                logger.warning(
                    f"Non-finite loss ({loss_val}) detected; skipping backward "
                    "to prevent model corruption."
                )
                del g, targets, outputs, loss
                return loss_val
            scaler.scale(loss).backward()
            if config.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model_instance.parameters(), config.max_grad_norm
                )
            scaler.step(optimizer)
            scaler.update()
            del g, targets, outputs, loss
            return loss_val
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if "out of memory" not in str(exc).lower():
                raise
            optimizer.zero_grad(set_to_none=True)
            # Do NOT call scaler.update() here: step() was never reached, so
            # the scaler has no inf checks recorded and update() would raise.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
            _oom_skipped[0] += 1
            logger.warning(
                f"OOM on batch (skipped {_oom_skipped[0]} total); "
                "GPU cache cleared, continuing training."
            )
            return float("nan")

    if deterministic:
        from ignite.engine import DeterministicEngine
        trainer = DeterministicEngine(_oom_safe_process_fn)
    else:
        from ignite.engine import Engine
        trainer = Engine(_oom_safe_process_fn)

    # LR scheduler step after each iteration
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    # Checkpointing
    if config.write_checkpoint:
        # Do NOT include 'trainer' in to_save: it would create a reference cycle
        # (trainer → Checkpoint handler → to_save["trainer"] → trainer) that
        # prevents the optimizer's GPU tensors from being GC-collected between fits.
        to_save = {
            "model": model_instance,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
        handler = Checkpoint(
            to_save,
            DiskSaver(config.output_dir, create_dir=True, require_empty=False),
            n_saved=2,
            global_step_transform=lambda *_: trainer.state.epoch,
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

    if config.progress:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})

    # --- evaluators ---
    train_evaluator = create_supervised_evaluator(
        model_instance,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=model_instance.device,
    )
    val_evaluator = create_supervised_evaluator(
        model_instance,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=model_instance.device,
    )

    history = {
        "train": {m: [] for m in metrics},
        "val": {m: [] for m in metrics},
    }

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        train_evaluator.run(train_loader)
        val_evaluator.run(val_loader)

        tm = train_evaluator.state.metrics
        vm = val_evaluator.state.metrics

        # Release last-batch tensors retained in engine state after each eval pass.
        train_evaluator.state.output = None
        val_evaluator.state.output = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        for metric in metrics:
            for src, store in ((tm, "train"), (vm, "val")):
                val = src[metric]
                if isinstance(val, torch.Tensor):
                    val = val.cpu().numpy().tolist()
                history[store][metric].append(val)

        if config.progress:
            pbar = ProgressBar()
            pbar.log_message(
                f"Train MAE: {tm['mae']:.4f}  RMSE: {tm['rmse']:.4f}"
            )
            pbar.log_message(
                f"  Val MAE: {vm['mae']:.4f}  RMSE: {vm['rmse']:.4f}"
            )

    # --- early stopping ---
    if config.early_stopping_patience > 0:
        # For regression: lower val loss is better (negate for the score function).
        # For classification: higher val ROC AUC is better.
        if config.classification_threshold is not None:
            def _score_fn(engine):
                return engine.state.metrics["rocauc"]
        else:
            def _score_fn(engine):
                return -(engine.state.metrics["loss"])

        es_handler = EarlyStopping(
            patience=config.early_stopping_patience,
            score_function=_score_fn,
            trainer=trainer,
            min_delta=config.early_stopping_min_delta,
        )
        val_evaluator.add_event_handler(Events.COMPLETED, es_handler)
        logger.info(
            f"Early stopping enabled: patience={config.early_stopping_patience}, "
            f"min_delta={config.early_stopping_min_delta}"
        )

    return trainer


# =============================================================================
# ALIGNN Model Class
# =============================================================================

class AlignnLayerNorm(ALIGNN):
    """ALIGNN with LayerNorm, providing a scikit-learn compatible interface.

    Inherits from nfflr.models.ALIGNN and overrides norm to "layernorm".
    """

    def __init__(
        self,
        config: NfflrAlignnConfig,
        chk_file: Optional[str] = None,
        reset_parameters: bool = True,
    ):
        alignn_cfg = config.to_alignn_config()
        alignn_cfg.norm = "layernorm"
        super().__init__(alignn_cfg)
        _init_model(self, config, chk_file, reset_parameters)

    def fit(
        self,
        X: Union[pd.DataFrame, pd.Series],
        y: Union[pd.DataFrame, pd.Series],
        val: Optional[Tuple] = None,
    ) -> "AlignnLayerNorm":
        """Train on (X, y); optionally evaluate on val each epoch.

        Args:
            X: Training atom structures (jarvis Atoms dicts, nfflr.Atoms, or pymatgen Structures)
            y: Training targets
            val: Optional (X_val, y_val) tuple for validation

        Returns:
            self (fitted model)
        """
        _fit_model(self, X, y, val=val)
        return self

    def predict(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Return predictions as a Series indexed by sample ID.

        Args:
            X: Atom structures to predict on

        Returns:
            Predictions as pandas Series
        """
        return _predict_with_model(self, X)
