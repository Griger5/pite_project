#!/usr/bin/env python3
import argparse
import csv
import random
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from AI_dio.data_preprocessing.audio_utils import load_audio_mono_resampled
from AI_dio.data_preprocessing.dataset import AIDetectDataset
from AI_dio.data_preprocessing.features import (
    FeatureParams,
    build_mel_transforms,
    mel_tokens_from_audio,
)
from AI_dio.training.checkpoints import load_checkpoint
from AI_dio.training.common import (
    choose_device,
    collate_fn,
    get_section,
    load_yaml_config,
    resolve_path,
)
from AI_dio.training.metrics import BinaryMetricsAccumulator
from AI_dio.training.models import BaselineCNN

ROOT = Path(__file__).parents[1].resolve()

CODEC_SETTINGS = {
    "mp3": {"ext": ".mp3", "args": ["-c:a", "libmp3lame", "-q:a", "5"]},
    "aac": {"ext": ".m4a", "args": ["-c:a", "aac", "-b:a", "128k"]},
    "opus": {"ext": ".opus", "args": ["-c:a", "libopus", "-b:a", "64k"]},
}


class AudioFeatureDataset(Dataset):
    def __init__(self, rows: list[dict], params: FeatureParams):
        self.rows = rows
        self.params = params
        self._target_length = int(params.chunk_duration * params.target_sr)
        self._mel, self._to_db = build_mel_transforms(params)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = row["path"]
        y = int(row["label"])
        audio = load_audio_mono_resampled(
            path, target_sr=self.params.target_sr, target_length=self._target_length
        )
        tokens = mel_tokens_from_audio(
            audio, self.params, mel=self._mel, to_db=self._to_db
        )
        if tokens.dim() == 3:
            tokens = tokens.squeeze(0)
        return tokens, y


def load_manifest_rows(manifest_path: Path, split: str) -> list[dict]:
    with open(manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return [r for r in reader if r.get("split") == split]


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    crit = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    correct = 0
    correct_flip = 0
    running_loss = 0.0
    metrics_acc = BinaryMetricsAccumulator(track_pr_auc=True)
    for x, y in tqdm(loader, desc="eval"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(x)
            loss = crit(logits, y)
        batch = y.size(0)
        running_loss += loss.item() * batch
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        correct_flip += ((1 - preds) == y).sum().item()
        total += batch
        metrics_acc.update(logits, y)
    loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    acc_flip = correct_flip / max(total, 1)
    metrics = {"loss": loss, "acc": acc, "acc_flip": acc_flip}
    metrics.update(metrics_acc.compute())
    return metrics


def run_ffmpeg(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{result.stderr.strip()}")


def reencode_audio(
    src: Path,
    dst: Path,
    codec: str,
    sample_rate: int,
    ffmpeg_bin: str,
) -> None:
    settings = CODEC_SETTINGS[codec]
    tmp = dst.with_suffix(settings["ext"])
    encode_cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        *settings["args"],
        str(tmp),
    ]
    decode_cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(tmp),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    run_ffmpeg(encode_cmd)
    run_ffmpeg(decode_cmd)
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass


def build_stress_rows(
    rows: Iterable[dict],
    out_dir: Path,
    codecs: list[str],
    sample_rates: list[int],
    seed: int,
    max_items: int,
) -> tuple[list[dict], int]:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg not found in PATH.")

    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    stress_rows: list[dict] = []
    failures = 0
    for idx, row in enumerate(rows):
        if max_items and idx >= max_items:
            break
        src = Path(row["path"])
        codec = rng.choice(codecs)
        sample_rate = rng.choice(sample_rates)
        dst = out_dir / f"{idx:06d}_{codec}_{sample_rate}.wav"
        try:
            reencode_audio(src, dst, codec, sample_rate, ffmpeg_bin)
        except Exception:
            if codec != "mp3":
                try:
                    reencode_audio(src, dst, "mp3", sample_rate, ffmpeg_bin)
                except Exception:
                    failures += 1
                    continue
            else:
                failures += 1
                continue
        stress_rows.append({"path": str(dst), "label": row["label"]})
    return stress_rows, failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on test data.")
    parser.add_argument("--checkpoint", default="checkpoints/model_best.pt")
    parser.add_argument("--config", default="training_config.yml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--stress-dir", default="checkpoints/stress_tmp")
    parser.add_argument("--stress-max", type=int, default=0)
    parser.add_argument("--stress-seed", type=int, default=1337)
    parser.add_argument("--stress-codecs", default="mp3,aac,opus")
    parser.add_argument(
        "--stress-sr",
        default="8000,11025,12000,16000,22050,24000,32000,44100,48000",
    )
    args = parser.parse_args()

    config_path = resolve_path(ROOT, args.config)
    config = load_yaml_config(config_path)
    data_cfg = get_section(config, "data")
    loader_cfg = get_section(config, "loader")

    device = choose_device(args.device)
    print("Using", device)

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else loader_cfg.get("batch_size", 256)
    )
    num_workers = args.num_workers
    pin_memory = loader_cfg.get("pin_memory", True)
    persistent_workers = loader_cfg.get("persistent_workers", False)
    prefetch_factor = loader_cfg.get("prefetch_factor", 2)

    manifest = resolve_path(ROOT, data_cfg.get("manifest") or "manifest.csv")
    use_cache = data_cfg.get("use_cache", True) and not args.no_cache
    cache_dir = data_cfg.get("cache_dir") or "data/cache/mel_3s_16k"
    cache_dir = resolve_path(ROOT, cache_dir) if use_cache else None

    checkpoint_path = resolve_path(ROOT, args.checkpoint)
    model = BaselineCNN()
    ckpt = load_checkpoint(model, checkpoint_path, device)
    if isinstance(ckpt, dict):
        print("Loaded checkpoint epoch:", ckpt.get("epoch"))
        print("Best metric:", ckpt.get("best_metric"))

    params = FeatureParams()
    if use_cache:
        dataset = AIDetectDataset(
            str(manifest),
            split=args.split,
            cache_dir=str(cache_dir),
            chunk_duration=params.chunk_duration,
            target_sr=params.target_sr,
            win_ms=params.win_ms,
            hop_ms=params.hop_ms,
            n_mels=params.n_mels,
        )
    else:
        rows = load_manifest_rows(manifest, args.split)
        dataset = AudioFeatureDataset(rows, params)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
        "drop_last": False,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    metrics = evaluate(model, loader, device)
    print(
        "test loss={loss:.4f} acc={acc:.4f} acc_flipped={acc_flip:.4f} (n={n})".format(
            loss=metrics["loss"],
            acc=metrics["acc"],
            acc_flip=metrics["acc_flip"],
            n=len(dataset),
        )
    )
    if "recall0" in metrics:
        print(
            "class0: recall={recall0:.4f} precision={precision0:.4f} f1={f1_0:.4f}".format(
                recall0=metrics["recall0"],
                precision0=metrics["precision0"],
                f1_0=metrics["f1_0"],
            )
        )
        print(
            "balanced_acc={balanced_acc:.4f} recall1={recall1:.4f}".format(
                balanced_acc=metrics["balanced_acc"],
                recall1=metrics["recall1"],
            )
        )
        print(
            "pr_auc(pos=1)={pr_auc_1:.4f} pr_auc(pos=0)={pr_auc_0:.4f}".format(
                pr_auc_1=metrics["pr_auc_1"],
                pr_auc_0=metrics["pr_auc_0"],
            )
        )
        print(
            "confusion(class0_positive): TP0={tp0} FN0={fn0} FP0={fp0} TN0={tn0}".format(
                tp0=metrics["tp0"],
                fn0=metrics["fn0"],
                fp0=metrics["fp0"],
                tn0=metrics["tn0"],
            )
        )

    if args.stress:
        rows = load_manifest_rows(manifest, args.split)
        codecs = [c.strip() for c in args.stress_codecs.split(",") if c.strip()]
        codecs = [c for c in codecs if c in CODEC_SETTINGS]
        if not codecs:
            raise ValueError("No valid codecs provided for stress test.")
        sample_rates = [
            int(sr) for sr in args.stress_sr.split(",") if sr.strip().isdigit()
        ]
        if not sample_rates:
            raise ValueError("No valid sample rates provided for stress test.")

        stress_dir = resolve_path(ROOT, args.stress_dir)
        stress_rows, failures = build_stress_rows(
            rows,
            stress_dir,
            codecs,
            sample_rates,
            seed=args.stress_seed,
            max_items=args.stress_max,
        )
        if failures:
            print(f"Stress test skipped {failures} files due to ffmpeg errors.")
        if not stress_rows:
            raise RuntimeError("No stress-test audio could be generated.")

        stress_dataset = AudioFeatureDataset(stress_rows, params)
        stress_loader = DataLoader(stress_dataset, **loader_kwargs)
        metrics = evaluate(model, stress_loader, device)
        print(
            "stress loss={loss:.4f} acc={acc:.4f} acc_flipped={acc_flip:.4f} "
            "(n={n})".format(
                loss=metrics["loss"],
                acc=metrics["acc"],
                acc_flip=metrics["acc_flip"],
                n=len(stress_dataset),
            )
        )
        if "recall0" in metrics:
            print(
                "stress class0: recall={recall0:.4f} precision={precision0:.4f} "
                "f1={f1_0:.4f}".format(
                    recall0=metrics["recall0"],
                    precision0=metrics["precision0"],
                    f1_0=metrics["f1_0"],
                )
            )
            print(
                "stress balanced_acc={balanced_acc:.4f} recall1={recall1:.4f}".format(
                    balanced_acc=metrics["balanced_acc"],
                    recall1=metrics["recall1"],
                )
            )
            print(
                "stress pr_auc(pos=1)={pr_auc_1:.4f} pr_auc(pos=0)={pr_auc_0:.4f}".format(
                    pr_auc_1=metrics["pr_auc_1"],
                    pr_auc_0=metrics["pr_auc_0"],
                )
            )
            print(
                "stress confusion(class0_positive): TP0={tp0} FN0={fn0} FP0={fp0} "
                "TN0={tn0}".format(
                    tp0=metrics["tp0"],
                    fn0=metrics["fn0"],
                    fp0=metrics["fp0"],
                    tn0=metrics["tn0"],
                )
            )
