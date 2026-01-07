import argparse
import json
import uuid
from pathlib import Path

from AI_dio.training.checkpoints import load_checkpoint_payload

ROOT = Path(__file__).resolve().parents[1]


def _rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_metadata(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return list(data.values())
    raise ValueError(f"Unsupported metadata format in {path}")


def _save_metadata(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def _normalize_metrics(metrics: dict) -> dict:
    if not isinstance(metrics, dict):
        return {}
    normalized = {}
    for key, value in metrics.items():
        norm_key = key.replace("/", "_")
        normalized[norm_key] = value
    return normalized


def _extract_core_metrics(ckpt: dict) -> dict:
    if not isinstance(ckpt, dict):
        return {}
    metrics = _normalize_metrics(ckpt.get("metrics", {}))
    core = {}
    for key in ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"):
        if key in metrics:
            core[key] = metrics[key]
    if "epoch" not in core and "epoch" in ckpt:
        core["epoch"] = ckpt["epoch"]
    best = ckpt.get("best_metric")
    if isinstance(best, dict) and "name" in best and "value" in best:
        core["best_metric"] = {"name": best["name"], "value": best["value"]}
    return core


def _collect_checkpoints(checkpoints_dir: Path) -> list[Path]:
    if not checkpoints_dir.exists():
        return []
    return sorted(p for p in checkpoints_dir.glob("*.pt") if p.is_file())


def snapshot_checkpoint(
    checkpoint_path: Path, snapshots_dir: Path, metadata: list[dict]
) -> dict:
    ckpt = load_checkpoint_payload(checkpoint_path)
    core_metrics = _extract_core_metrics(ckpt)
    snap_id = uuid.uuid4().hex
    snapshot_path = snapshots_dir / f"{snap_id}.pt"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.replace(snapshot_path)
    entry = {
        "id": snap_id,
        "path": _rel_or_abs(snapshot_path),
        "metrics": core_metrics,
    }
    metadata.append(entry)
    return entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Snapshot checkpoints into a separate folder with metadata."
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=ROOT / "checkpoints",
        help="Directory containing .pt checkpoints to snapshot.",
    )
    parser.add_argument(
        "--snapshots-dir",
        type=Path,
        default=ROOT / "checkpoints" / "snapshots",
        help="Destination directory for snapshot files.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "checkpoints_metadata.json",
        help="Path to checkpoints metadata JSON file.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        default=[],
        help="Specific checkpoint file to snapshot (repeatable).",
    )
    args = parser.parse_args()

    checkpoints = [p.resolve() for p in args.checkpoint]
    if not checkpoints:
        checkpoints = _collect_checkpoints(args.checkpoints_dir)
    if not checkpoints:
        print("No checkpoints found.")
        exit(0)

    metadata = _load_metadata(args.metadata)
    for checkpoint_path in checkpoints:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        entry = snapshot_checkpoint(checkpoint_path, args.snapshots_dir, metadata)
        print(f"Snapshotted {checkpoint_path} -> {entry['path']} (id={entry['id']})")

    _save_metadata(args.metadata, metadata)
    print(f"Updated metadata: {_rel_or_abs(args.metadata)}")
