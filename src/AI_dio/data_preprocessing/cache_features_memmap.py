import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from AI_dio.data_preprocessing.audio_utils import load_audio_mono_resampled
from AI_dio.data_preprocessing.augmentations import AudioAugmenter
from AI_dio.data_preprocessing.features import (
    FeatureParams,
    build_mel_transforms,
    mel_tokens_from_audio,
    num_frames,
    stft_params,
)

ROOT = Path(__file__).parents[3].resolve()


def _load_manifest(manifest_csv: Path) -> dict[str, list[dict]]:
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    with open(manifest_csv) as f:
        reader = csv.DictReader(f.readlines(), delimiter=",")
        for row in reader:
            split = row.get("split")
            if split in splits:
                splits[split].append(row)
    return splits


class _AudioDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        target_sr: int,
        chunk_length: int,
        *,
        augment: bool = False,
        augment_root: Path | None = None,
        augment_cfg: dict | None = None,
    ) -> None:
        self._rows = rows
        self._target_sr = target_sr
        self._chunk_length = chunk_length
        self._augmenter: AudioAugmenter | None = None

        if augment:
            if augment_root is None:
                raise ValueError("augment_root is required when augment=True")
            cfg = augment_cfg or {}
            self._augmenter = AudioAugmenter(
                augment_root=augment_root,
                target_sr=target_sr,
                chunk_length=chunk_length,
                p_noise=float(cfg.get("p_noise", 0.6)),
                p_music=float(cfg.get("p_music", 0.3)),
                p_rir=float(cfg.get("p_rir", 0.3)),
                snr_db_min=float(cfg.get("snr_db_min", -5.0)),
                snr_db_max=float(cfg.get("snr_db_max", 15.0)),
                music_snr_db_min=float(cfg.get("music_snr_db_min", -10.0)),
                music_snr_db_max=float(cfg.get("music_snr_db_max", 10.0)),
                gain_db_min=float(cfg.get("gain_db_min", -6.0)),
                gain_db_max=float(cfg.get("gain_db_max", 6.0)),
                allow_music_and_noise=bool(cfg.get("allow_music_and_noise", False)),
                preload=bool(cfg.get("preload", False)),
                preload_max_files=int(cfg.get("preload_max_files", 256)),
                preload_segments_per_file=int(cfg.get("preload_segments_per_file", 1)),
            )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor, int, bool]:
        row = self._rows[idx]
        ok = True
        try:
            audio_tensor = load_audio_mono_resampled(
                row["path"],
                target_sr=self._target_sr,
                target_length=self._chunk_length,
            )
        except Exception:
            ok = False
            audio_tensor = torch.zeros((1, self._chunk_length), dtype=torch.float32)
        if ok and self._augmenter is not None:
            audio_tensor = self._augmenter.apply(audio_tensor)
        return idx, audio_tensor, int(row["label"]), ok


def _write_split_cache(
    *,
    split: str,
    rows: list[dict],
    output_dir: Path,
    mel,
    to_db,
    chunk_length: int,
    num_frames: int,
    n_mels: int,
    dtype: np.dtype,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int | None,
    params: FeatureParams,
    augment: bool,
    augment_root: Path | None,
    augment_cfg: dict | None,
) -> dict:
    num_samples = len(rows)
    features_path = output_dir / f"features_{split}.mmap"
    labels_path = output_dir / f"labels_{split}.npy"

    features = np.memmap(
        features_path,
        mode="w+",
        dtype=dtype,
        shape=(num_samples, num_frames, n_mels),
    )
    labels = np.empty((num_samples,), dtype=np.int64)

    dataset = _AudioDataset(
        rows=rows,
        target_sr=params.target_sr,
        chunk_length=chunk_length,
        augment=augment,
        augment_root=augment_root,
        augment_cfg=augment_cfg,
    )
    loader_kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)
    non_blocking = pin_memory and device.type == "cuda"

    with torch.inference_mode():
        failed = 0
        for indices, audio_batch, labels_batch, ok_batch in tqdm(
            loader, desc=f"Cache {split}"
        ):
            ok_tensor = torch.as_tensor(ok_batch)
            failed += int((~ok_tensor).sum().item())
            audio_batch = audio_batch.to(device, non_blocking=non_blocking)
            if audio_batch.dim() == 3 and audio_batch.size(1) == 1:
                audio_batch = audio_batch.squeeze(1)
            tokens = mel_tokens_from_audio(audio_batch, params, mel=mel, to_db=to_db)
            if tokens.dim() != 3:
                raise RuntimeError(
                    f"Unexpected mel shape for split '{split}': {tuple(tokens.shape)}"
                )

            if tokens.shape[1] != num_frames:
                if tokens.shape[1] > num_frames:
                    tokens = tokens[:, :num_frames, :]
                else:
                    pad = num_frames - tokens.shape[1]
                    tokens = torch.nn.functional.pad(tokens, (0, 0, 0, pad))

            tokens_np = tokens.cpu().numpy().astype(dtype, copy=False)
            indices_np = indices.cpu().numpy()
            labels_np = labels_batch.cpu().numpy()
            if indices_np.size > 0:
                start = int(indices_np[0])
                if np.array_equal(
                    indices_np, np.arange(start, start + indices_np.size)
                ):
                    features[start : start + indices_np.size] = tokens_np
                    labels[start : start + indices_np.size] = labels_np
                else:
                    for row_idx, sample_idx in enumerate(indices_np):
                        features[int(sample_idx)] = tokens_np[row_idx]
                        labels[int(sample_idx)] = int(labels_np[row_idx])

    features.flush()
    np.save(labels_path, labels)
    if failed:
        print(
            f"[warn] {failed} audio files failed to decode in split '{split}'. "
            "Silence was cached for those entries."
        )

    return {
        "num_samples": num_samples,
        "features": features_path.name,
        "labels": labels_path.name,
    }


def build_cache(
    manifest_csv: Path,
    output_dir: Path,
    chunk_duration: float,
    target_sr: int,
    win_ms: float,
    hop_ms: float,
    n_mels: int,
    dtype: np.dtype,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int | None,
    augment: bool,
    augment_root: Path | None,
    augment_cfg: dict | None,
    augment_splits: set[str],
) -> Path:
    splits = _load_manifest(manifest_csv)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        raise FileExistsError(f"Cache already exists: {metadata_path}")

    params = FeatureParams(
        chunk_duration=chunk_duration,
        target_sr=target_sr,
        win_ms=win_ms,
        hop_ms=hop_ms,
        n_mels=n_mels,
    )
    win_length, hop_length, n_fft = stft_params(params)
    chunk_length = int(params.chunk_duration * params.target_sr)
    frames_per_clip = num_frames(params, center=True)

    mel, to_db = build_mel_transforms(params, device=device)
    mel.eval()
    to_db.eval()

    metadata = {
        "chunk_duration": chunk_duration,
        "target_sr": target_sr,
        "win_ms": win_ms,
        "hop_ms": hop_ms,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "chunk_length": chunk_length,
        "num_frames": frames_per_clip,
        "dtype": str(dtype),
        "splits": {},
    }

    for split, rows in splits.items():
        if not rows:
            continue
        split_meta = _write_split_cache(
            split=split,
            rows=rows,
            output_dir=output_dir,
            mel=mel,
            to_db=to_db,
            chunk_length=chunk_length,
            num_frames=frames_per_clip,
            n_mels=params.n_mels,
            dtype=dtype,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            params=params,
            augment=augment and split in augment_splits,
            augment_root=augment_root,
            augment_cfg=augment_cfg,
        )
        metadata["splits"][split] = split_meta

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cache log-mel features into per-split memmap files."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)

    args = parser.parse_args()
    dtype = np.dtype("float32")
    if dtype.kind != "f":
        raise ValueError(f"Expected a float dtype, got {dtype}")
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if args.num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    device = torch.device("cpu")
    pin_memory = args.pin_memory and device.type == "cuda"
    prefetch_factor = args.prefetch_factor if args.num_workers > 0 else None

    augment_cfg = None
    augment_enabled = False
    augment_root = None
    augment_splits = {"train"}

    metadata_path = build_cache(
        manifest_csv=ROOT / "manifest.csv",
        output_dir=ROOT / "data" / "cache" / "mel_3s_16k",
        chunk_duration=3.0,
        target_sr=16000,
        win_ms=25.0,
        hop_ms=10.0,
        n_mels=80,
        dtype=dtype,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        augment=augment_enabled,
        augment_root=augment_root,
        augment_cfg=augment_cfg,
        augment_splits=augment_splits,
    )
    print(f"Wrote cache metadata to {metadata_path}")
