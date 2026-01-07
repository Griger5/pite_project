from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import wandb
from AI_dio.data_preprocessing.dataset import AIDetectDataset
from AI_dio.training.checkpoints import save_checkpoint
from AI_dio.training.common import (
    choose_device,
    collate_fn,
    get_section,
    is_better_metric,
    resolve_metric,
    resolve_optional_path,
    resolve_path,
)
from AI_dio.training.models import BaselineCNN

ROOT = Path(__file__).parents[3].resolve()
DEFAULT_CKPT_DIR = ROOT / "checkpoints"


def init_wandb(wandb_cfg: dict, config: dict, config_path: Path):
    if not wandb_cfg.get("enabled", False):
        return None
    run = wandb.init(
        project=wandb_cfg.get("project", "AI_dio"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("run_name"),
        mode=wandb_cfg.get("mode"),
        config=config,
    )
    wandb.config.update({"config_path": str(config_path)}, allow_val_change=True)
    return run


def build_loader(
    manifest: Path,
    split: Literal["train", "val", "test"],
    cache_dir: Optional[Path],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    drop_last: bool,
    balanced_sampler: bool,
) -> DataLoader:
    dataset = AIDetectDataset(
        str(manifest),
        split,
        cache_dir=(None if cache_dir is None else str(cache_dir)),
    )
    sampler = None
    shuffle = split == "train"
    if split == "train" and balanced_sampler:
        counts = {0: 0, 1: 0}
        labels = []
        for row in dataset.rows:
            try:
                label = int(row["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid label in manifest: {row.get('label')}"
                ) from exc
            if label not in counts:
                raise ValueError(f"Unexpected label in training data: {label}")
            counts[label] += 1
            labels.append(label)
        if counts[0] == 0 or counts[1] == 0:
            raise ValueError(
                f"Balanced sampler requires both classes, got counts={counts}"
            )
        weights = [1.0 / counts[label] for label in labels]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        shuffle = False
    if num_workers > 0:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor,
            drop_last=drop_last if split == "train" else False,
            collate_fn=collate_fn,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        drop_last=drop_last if split == "train" else False,
        collate_fn=collate_fn,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    crit: nn.Module,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    epochs: int,
    clip_grad_norm: float,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [train]")
    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(x)
            loss = crit(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        if clip_grad_norm and clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(opt)
        scaler.update()

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += batch_size

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    crit: nn.Module,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc=f"Epoch {epoch}/{epochs} [val]"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(x)
            loss = crit(logits, y)

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += batch_size

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def run_training(config: dict, config_path: Path) -> None:
    data_cfg = get_section(config, "data")
    loader_cfg = get_section(config, "loader")
    train_cfg = get_section(config, "train")
    optim_cfg = get_section(config, "optim")
    metrics_cfg = get_section(config, "metrics")
    wandb_cfg = get_section(config, "wandb")
    ckpt_cfg = get_section(config, "checkpoints")

    seed = train_cfg.get("seed", 1337)
    if seed is not None:
        torch.manual_seed(seed)

    device_name = train_cfg.get("device", "auto")
    device = choose_device(device_name)

    print("Using", device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    use_cache = data_cfg.get("use_cache", True)
    manifest = resolve_path(ROOT, data_cfg.get("manifest") or "manifest.csv")
    cache_dir = resolve_optional_path(
        ROOT,
        (data_cfg.get("cache_dir") or "data/cache/mel_3s_16k") if use_cache else None,
    )

    batch_size = loader_cfg.get("batch_size", 256)
    num_workers = loader_cfg.get("num_workers", 8)
    pin_memory = loader_cfg.get("pin_memory", True)
    persistent_workers = loader_cfg.get("persistent_workers", True)
    prefetch_factor = loader_cfg.get("prefetch_factor", 4)
    drop_last = loader_cfg.get("drop_last", True)
    balanced_sampler = loader_cfg.get("balanced_sampler", True)

    epochs = train_cfg.get("epochs", 10)
    clip_grad_norm = train_cfg.get("clip_grad_norm", 1.0)
    metrics_every = metrics_cfg.get("every", 1)
    save_best = ckpt_cfg.get("save_best", True)
    save_last = ckpt_cfg.get("save_last", True)
    best_metric = ckpt_cfg.get("metric", "val_loss")
    ckpt_dir = resolve_path(ROOT, ckpt_cfg.get("dir", DEFAULT_CKPT_DIR))

    lr = optim_cfg.get("lr", 3e-4)
    weight_decay = optim_cfg.get("weight_decay", 1e-2)

    train_loader = build_loader(
        manifest=manifest,
        split="train",
        cache_dir=cache_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        balanced_sampler=balanced_sampler,
    )
    val_loader = None
    try:
        val_loader = build_loader(
            manifest=manifest,
            split="val",
            cache_dir=cache_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=False,
            balanced_sampler=False,
        )
        if len(val_loader) == 0:
            val_loader = None
    except Exception:
        val_loader = None

    model = BaselineCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    wandb_run = init_wandb(wandb_cfg, config, config_path)
    best_metric_key = None
    best_metric_value = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            crit=crit,
            opt=opt,
            scaler=scaler,
            device=device,
            epoch=epoch,
            epochs=epochs,
            clip_grad_norm=clip_grad_norm,
        )

        metrics = {
            "train/loss": train_loss,
            "train/acc": train_acc,
            "epoch": epoch,
            "lr": opt.param_groups[0]["lr"],
        }
        should_log = metrics_every > 0 and (
            epoch % metrics_every == 0 or epoch == epochs
        )
        should_eval = val_loader is not None and (
            should_log
            or (save_best and best_metric.replace("/", "_").startswith("val_"))
        )
        if should_eval:
            val_loss, val_acc = evaluate(
                model=model,
                loader=val_loader,
                crit=crit,
                device=device,
                epoch=epoch,
                epochs=epochs,
            )
            metrics["val/loss"] = val_loss
            metrics["val/acc"] = val_acc

        if should_log:
            print(
                " | ".join(
                    [
                        f"epoch={epoch}/{epochs}",
                        f"train_loss={metrics['train/loss']:.4f}",
                        f"train_acc={metrics['train/acc']:.4f}",
                        (
                            f"val_loss={metrics['val/loss']:.4f}"
                            if "val/loss" in metrics
                            else "val_loss=NA"
                        ),
                        (
                            f"val_acc={metrics['val/acc']:.4f}"
                            if "val/acc" in metrics
                            else "val_acc=NA"
                        ),
                    ]
                )
            )

            if wandb_run is not None:
                wandb_run.log(metrics, step=epoch)

        if save_best or save_last:
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "scaler_state": scaler.state_dict(),
                "metrics": metrics,
                "config_path": str(config_path),
            }

            if save_last:
                save_checkpoint(ckpt_dir / "model_last.pt", checkpoint)

            if save_best:
                metric_key, metric_value = resolve_metric(metrics, best_metric)
                if is_better_metric(metric_key, metric_value, best_metric_value):
                    best_metric_key = metric_key
                    best_metric_value = metric_value
                    checkpoint["best_metric"] = {
                        "name": best_metric_key,
                        "value": best_metric_value,
                    }
                    save_checkpoint(ckpt_dir / "model_best.pt", checkpoint)

    if wandb_run is not None:
        wandb_run.finish()
