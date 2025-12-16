import csv
import os
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

ROOT = Path(__file__).parents[2].resolve()
RAW_DATA_DIR = ROOT / "data" / "raw"
FoR_DIR = RAW_DATA_DIR / "for-norm"


def group_id_from_filename_for(fn: Path) -> str:
    name = fn.name
    suffix = ".wav_16k.wav_norm.wav_mono.wav_silence.wav"
    if name.endswith(suffix):
        return name[: -len(suffix)] + ".wav"
    if ".wav_" in name:
        return name.split(".wav_", 1)[0] + ".wav"
    return fn.stem


def determine_cls_from_filepath_for(fp: Path) -> int:
    if any(p.lower() == "real" for p in fp.parts):
        return 0
    if any(p.lower() == "fake" for p in fp.parts):
        return 1
    raise ValueError(f"Can't infer label (real/fake) from path: {fp}")


def build_manifest(
    manifest_path: Path,
    source_name: str,
    training_pattern: str,
    testing_pattern: str,
    valdiation_pattern: str,
    group_id_from_name: Callable[[Path], str],
    determine_cls_from_filepath: Callable[[Path], int],
):
    out_path = Path(manifest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def expand_pattern(pattern: str) -> list[Path]:
        return [p for p in Path().glob(pattern)]

    def matches_are_wav(matches: list[Path]) -> bool:
        return any(m.is_file() and m.suffix.lower() == ".wav" for m in matches)

    def rows_from_wav_files(matches: list[Path]) -> Iterable[dict]:
        for p in matches:
            if not (p.is_file() and p.suffix.lower() == ".wav"):
                continue
            label = determine_cls_from_filepath(p)

            yield {
                "path": str(p.resolve()),
                "label": str(label),
                "split": split_name,
                "source": source_name,
                "group_id": group_id_from_name(p),
            }

    def rows_from_dirs(root: Path) -> Iterable[dict]:
        for cls_dir, label in [("real", 0), ("fake", 1)]:
            base = root / cls_dir
            if not base.exists():
                continue
            for p in base.rglob("*.wav"):
                yield {
                    "path": str(p.resolve()),
                    "label": str(label),
                    "split": split_name,
                    "source": source_name,
                    "group_id": group_id_from_name(p),
                }

    def rows_from_pattern(pattern: str) -> Iterable[dict]:
        matches = expand_pattern(pattern)
        if not matches:
            raise FileNotFoundError(f"Pattern matched nothing: {pattern}")

        any_wavs = matches_are_wav(matches)
        if any_wavs:
            for row in rows_from_wav_files(matches):
                yield row
            return

        for root in matches:
            if not root.exists():
                continue
            if root.is_file():
                continue

            for row in rows_from_dirs(root):
                yield row

    all_rows = []
    for split_name, pat in [
        ("train", training_pattern),
        ("val", valdiation_pattern),
        ("test", testing_pattern),
    ]:
        all_rows.extend(list(rows_from_pattern(pat)))

    seen = set()
    deduped = []
    for r in all_rows:
        if r["path"] in seen:
            continue
        seen.add(r["path"])
        deduped.append(r)
    all_rows = deduped

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["path", "label", "split", "source", "group_id"]
        )
        w.writeheader()
        w.writerows(all_rows)

    return out_path


def manifest_healthcheck(manifest_path: Path) -> str:
    issues: list[str] = []

    p = Path(manifest_path)
    if not p.exists():
        issues.append(f"Missing manifest: {p}")
    else:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            issues.append(f"Failed to read manifest: {e}")
        else:
            required = {"path", "label", "split"}
            missing_cols = required - set(df.columns)
            if missing_cols:
                issues.append(f"Missing columns: {sorted(missing_cols)}")

            if "label" in df.columns:
                bad_labels = sorted(set(df["label"].dropna().unique()) - {0, 1})
                if bad_labels:
                    issues.append(f"Invalid labels: {bad_labels}")

            if "path" in df.columns:
                missing_files = df[
                    ~df["path"].astype(str).map(lambda x: Path(x).exists())
                ]
                if len(missing_files) > 0:
                    issues.append(f"Missing audio files: {len(missing_files)}")

    if issues:
        return "FAIL\n" + "\n".join(issues)
    return "OK\n"


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


if __name__ == "__main__":
    if any((not os.path.exists(missing := p)) for p in [ROOT, RAW_DATA_DIR, FoR_DIR]):
        raise FileNotFoundError(
            f"Missing file or directory: {missing}. Please refer to {ROOT / 'README.md'} for details"
        )
    manifest_fp = ROOT / "manifest.csv"

    build_manifest(
        manifest_fp,
        "FoR-norm",
        training_pattern="./data/raw/for-norm/training/**/*.wav",
        testing_pattern="./data/raw/for-norm/testing/**/*.wav",
        valdiation_pattern="./data/raw/for-norm/validation/**/*.wav",
        group_id_from_name=group_id_from_filename_for,
        determine_cls_from_filepath=determine_cls_from_filepath_for,
    )
    print("Manifest has been built at:", manifest_fp)
    print("Running healthcheck:")
    print(manifest_healthcheck(manifest_fp))
    print("Manifest summary:")
    print(manifest_summary(manifest_fp))
