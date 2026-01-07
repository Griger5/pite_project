import argparse
import csv
from pathlib import Path

import pandas as pd
import torchaudio

ROOT = Path(__file__).resolve().parents[1]


def audio_is_decodable(p: Path) -> bool:
    try:
        if p.stat().st_size < 64:
            return False
    except OSError:
        return False

    try:
        audio_tensor, sr = torchaudio.load(str(p))
        return audio_tensor.numel() > 0 and sr > 0
    except Exception:
        return False


def manifest_healthcheck(manifest_path: Path) -> str:
    p = Path(manifest_path)

    if not p.exists():
        return f"FAIL\nMissing manifest: {p}\n"

    try:
        df = pd.read_csv(p)
    except Exception as e:
        return f"FAIL\nFailed to read manifest: {e}\n"

    issues: list[str] = []

    required = {"path", "label", "split"}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    cols = set(df.columns)

    if "label" in cols:
        allowed = {0, 1, -1}
        bad = sorted(set(df["label"].dropna().unique()) - allowed)
        if bad:
            issues.append(f"Invalid labels: {bad}")

        if {"split", "label"} <= cols:
            bad_unknown = df[(df["label"] == -1) & (df["split"] != "test")]
            if not bad_unknown.empty:
                issues.append("Unknown labels (-1) are only allowed in test split")

    if "path" in cols:
        missing_files = ~df["path"].astype(str).map(lambda x: Path(x).exists())
        n_missing = int(missing_files.sum())
        if n_missing:
            issues.append(f"Missing audio files: {n_missing}")

    return ("FAIL\n" + "\n".join(issues) + "\n") if issues else "OK\n"


def manifest_summary(manifest_path: Path) -> str:
    p = Path(manifest_path)
    parts: list[str] = []

    if not p.exists():
        parts.append(f"Missing manifest: {p}")
        return "\n".join(parts)

    df = pd.read_csv(p)

    parts.append(f"rows={len(df)}")
    if "split" in df.columns:
        parts.append(f"splits={sorted(df['split'].dropna().unique().tolist())}")
    if "label" in df.columns:
        parts.append(f"labels={sorted(df['label'].dropna().unique().tolist())}")

    if {"split", "label"} <= set(df.columns):
        parts.append("\ncounts(split,label):")
        parts.append(df.groupby(["split", "label"]).size().to_string())

    if "path" in df.columns:
        missing = (~df["path"].astype(str).map(lambda x: Path(x).exists())).sum()
        parts.append(f"\nmissing_files={int(missing)}")

    return "\n".join(parts)


def decodability_summary(manifest_path: Path) -> str:
    if not manifest_path.exists():
        return f"Missing manifest: {manifest_path}"

    total = 0
    missing = 0
    undecodable = 0

    with open(manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            raw_path = (row.get("path") or "").strip()
            if not raw_path:
                missing += 1
                continue
            path = Path(raw_path)
            if not path.exists():
                missing += 1
                continue
            if not audio_is_decodable(path):
                undecodable += 1

    return "\n".join(
        [
            f"rows={total}",
            f"missing_files={missing}",
            f"undecodable_files={undecodable}",
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manifest summary + healthcheck + decodability checks."
    )
    parser.add_argument(
        "--manifest",
        default=ROOT / "manifest.csv",
        type=Path,
        help="Path to manifest CSV.",
    )
    args = parser.parse_args()
    manifest_path = Path(args.manifest)

    print("Manifest summary:")
    print(manifest_summary(manifest_path))
    print("Healthcheck:")
    print(manifest_healthcheck(manifest_path))
    print("Decodability check:")
    print(decodability_summary(manifest_path))
