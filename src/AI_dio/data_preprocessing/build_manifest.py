import csv
import os
import random
from pathlib import Path
from typing import Callable, Iterable, Optional

from tqdm import tqdm

ROOT = Path(__file__).parents[3].resolve()
RAW_DATA_DIR = ROOT / "data" / "raw"
FoR_DIR = RAW_DATA_DIR / "for-norm"
CODECFAKE_PLUS_DIR = RAW_DATA_DIR / "CodecFakePlus"
CODECFAKE_PLUS_CORS_DIR = CODECFAKE_PLUS_DIR / "Codecfake_plus_CoRS"
CODECFAKE_PLUS_COSG_DIR = CODECFAKE_PLUS_DIR / "CoSG"
CODECFAKE_PLUS_CORS_LABELS = CODECFAKE_PLUS_DIR / "CoRS_labels.txt"
CODECFAKE_PLUS_COSG_LABELS = CODECFAKE_PLUS_DIR / "CoSG_labels.txt"
CODECFAKE_SPLIT_RATIO = (0.8, 0.1, 0.1)

AUDIO_EXTS = {".wav", ".flac"}


class ManifestBuilder:
    def __init__(self) -> None:
        self._rows: list[dict] = []
        self._codecfake_label_cache: dict[Path, dict[str, int]] = {}

    def group_id_from_filename_for(self, fn: Path) -> str:
        name = fn.name
        suffix = ".wav_16k.wav_norm.wav_mono.wav_silence.wav"
        if name.endswith(suffix):
            return name[: -len(suffix)] + ".wav"
        if ".wav_" in name:
            return name.split(".wav_", 1)[0] + ".wav"
        return fn.stem

    def determine_cls_from_filepath_for(self, fp: Path) -> int:
        if any(p.lower() == "real" for p in fp.parts):
            return 0
        if any(p.lower() == "fake" for p in fp.parts):
            return 1
        raise ValueError(f"Can't infer label (real/fake) from path: {fp}")

    def _normalize_codecfake_label(self, label: str) -> int:
        norm = label.strip().lower()
        if norm in {"bonafide", "real"}:
            return 0
        if norm in {"spoof", "fake"}:
            return 1
        raise ValueError(f"Unknown CodecFakePlus label: {label!r}")

    def _load_codecfake_label_map(self, labels_path: Path) -> dict[str, int]:
        labels_path = Path(labels_path).resolve()
        cached = self._codecfake_label_cache.get(labels_path)
        if cached is not None:
            return cached

        mapping: dict[str, int] = {}
        with open(labels_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                if labels_path.name == "CoRS_labels.txt":
                    if len(parts) < 3:
                        raise ValueError(
                            f"Malformed CoRS label at line {line_no}: {line!r}"
                        )
                    file_name = parts[1]
                    label = self._normalize_codecfake_label(parts[2])
                elif labels_path.name == "CoSG_labels.txt":
                    if len(parts) < 6:
                        raise ValueError(
                            f"Malformed CoSG label at line {line_no}: {line!r}"
                        )
                    file_name = parts[1]
                    if not file_name.endswith(".wav"):
                        file_name = f"{file_name}.wav"
                    label = self._normalize_codecfake_label(parts[-1])
                else:
                    raise ValueError(f"Unknown CodecFakePlus label file: {labels_path}")

                existing = mapping.get(file_name)
                if existing is not None and existing != label:
                    raise ValueError(
                        "Conflicting CodecFakePlus labels for "
                        f"{file_name!r}: {existing} vs {label}"
                    )
                mapping[file_name] = label

        self._codecfake_label_cache[labels_path] = mapping
        return mapping

    def _group_id_from_codecfake_cors_name(self, file_name: str) -> str:
        stem = Path(file_name).stem
        parts = stem.split("_")
        if len(parts) >= 2:
            return "_".join(parts[:2]) + ".wav"
        return f"{stem}.wav"

    def _group_id_from_codecfake_cosg_name(self, file_name: str) -> str:
        stem = Path(file_name).stem
        return f"{stem}.wav"

    def add_rows_from_codecfake_labels(
        self,
        audio_root: Path,
        labels_path: Path,
        *,
        split_name: str = "train",
        split_ratio: tuple[float, float, float] | None = None,
        split_seed: int | None = None,
        source_name: str,
        group_id_from_name: Callable[[str], str],
    ) -> None:
        label_map = self._load_codecfake_label_map(labels_path)
        items: list[tuple[Path, int, str]] = []
        for file_name, label in tqdm(
            label_map.items(), desc=f"{source_name}:scan", total=len(label_map)
        ):
            p = audio_root / file_name
            if not (p.is_file() and p.suffix.lower() in AUDIO_EXTS):
                continue
            group_id = group_id_from_name(file_name)
            items.append((p, int(label), group_id))

        if split_ratio is not None:
            if len(split_ratio) != 3:
                raise ValueError("split_ratio must be a 3-tuple (train,val,test).")
            if any(r < 0 for r in split_ratio):
                raise ValueError("split_ratio values must be >= 0.")
            total_ratio = sum(split_ratio)
            if abs(total_ratio - 1.0) > 1e-6:
                raise ValueError("split_ratio must sum to 1.0.")

            groups: dict[str, list[tuple[Path, int, str]]] = {}
            for p, label, group_id in items:
                groups.setdefault(group_id, []).append((p, label, group_id))

            rng = (
                random.Random(split_seed) if split_seed is not None else random.Random()
            )
            group_ids = sorted(groups.keys())
            rng.shuffle(group_ids)
            n_total = len(group_ids)
            n_train = int(n_total * split_ratio[0])
            n_val = int(n_total * split_ratio[1])
            train_g = set(group_ids[:n_train])
            val_g = set(group_ids[n_train : n_train + n_val])

            for group_id in tqdm(group_ids, desc=f"{source_name}:split"):
                if group_id in train_g:
                    split_name = "train"
                elif group_id in val_g:
                    split_name = "val"
                else:
                    split_name = "test"
                for p, label, _ in groups[group_id]:
                    self._rows.append(
                        {
                            "path": str(p.resolve()),
                            "label": str(label),
                            "split": split_name,
                            "source": source_name,
                            "group_id": group_id,
                        }
                    )
            return

        for p, label, group_id in items:
            self._rows.append(
                {
                    "path": str(p.resolve()),
                    "label": str(label),
                    "split": split_name,
                    "source": source_name,
                    "group_id": group_id,
                }
            )

    def _expand_pattern(self, pattern: str) -> list[Path]:
        return [p for p in Path().glob(pattern)]

    def _matches_are_audio(self, matches: list[Path]) -> bool:
        return any(m.is_file() and m.suffix.lower() in AUDIO_EXTS for m in matches)

    def _rows_from_wav_files(
        self,
        matches: list[Path],
        *,
        source_name: str,
        split_name: str,
        group_id_from_name: Callable[[Path], str],
        determine_cls_from_filepath: Callable[[Path], int],
    ) -> Iterable[dict]:
        for p in tqdm(matches, desc=f"{source_name}:{split_name}"):
            if not (p.is_file() and p.suffix.lower() in AUDIO_EXTS):
                continue

            label = determine_cls_from_filepath(p)
            yield {
                "path": str(p.resolve()),
                "label": str(label),
                "split": split_name,
                "source": source_name,
                "group_id": group_id_from_name(p),
            }

    def _rows_from_dirs(
        self,
        root: Path,
        *,
        source_name: str,
        split_name: str,
        group_id_from_name: Callable[[Path], str],
    ) -> Iterable[dict]:
        for cls_dir, label in [("real", 0), ("fake", 1)]:
            base = root / cls_dir
            if not base.exists():
                continue
            for p in tqdm(
                (q for q in base.rglob("*") if q.suffix.lower() in AUDIO_EXTS),
                desc=f"{source_name}:{split_name}:{root.name}/{cls_dir}",
            ):
                yield {
                    "path": str(p.resolve()),
                    "label": str(label),
                    "split": split_name,
                    "source": source_name,
                    "group_id": group_id_from_name(p),
                }

    def _rows_from_pattern(
        self,
        pattern: str,
        *,
        source_name: str,
        split_name: str,
        group_id_from_name: Callable[[Path], str],
        determine_cls_from_filepath: Callable[[Path], int],
    ) -> Iterable[dict]:
        matches = self._expand_pattern(pattern)
        if not matches:
            raise FileNotFoundError(f"Pattern matched nothing: {pattern}")

        if self._matches_are_audio(matches):
            yield from self._rows_from_wav_files(
                matches,
                source_name=source_name,
                split_name=split_name,
                group_id_from_name=group_id_from_name,
                determine_cls_from_filepath=determine_cls_from_filepath,
            )
            return

        for root in matches:
            if not root.exists() or root.is_file():
                continue
            yield from self._rows_from_dirs(
                root,
                source_name=source_name,
                split_name=split_name,
                group_id_from_name=group_id_from_name,
            )

    def add_rows_from_patterns(
        self,
        source_name: str,
        training_pattern: str,
        testing_pattern: str,
        validation_pattern: str,
        group_id_from_name: Callable[[Path], str],
        determine_cls_from_filepath: Callable[[Path], int],
    ) -> None:
        for split_name, pat in [
            ("train", training_pattern),
            ("val", validation_pattern),
            ("test", testing_pattern),
        ]:
            for row in self._rows_from_pattern(
                pat,
                source_name=source_name,
                split_name=split_name,
                group_id_from_name=group_id_from_name,
                determine_cls_from_filepath=determine_cls_from_filepath,
            ):
                self._rows.append(row)

    def _dedupe_rows(self, rows: list[dict]) -> list[dict]:
        seen = set()
        deduped = []
        for r in rows:
            if r["path"] in seen:
                continue
            seen.add(r["path"])
            deduped.append(r)
        return deduped

    def _enforce_group_disjoint(self, rows: list[dict]) -> list[dict]:
        by_split = {"train": [], "val": [], "test": []}
        for r in rows:
            s = r["split"]
            if s in by_split:
                by_split[s].append(r)

        def gids(rs):
            return set(r["group_id"] for r in rs)

        test_g = gids(by_split["test"])

        before = len(by_split["val"])
        by_split["val"] = [r for r in by_split["val"] if r["group_id"] not in test_g]
        dropped_val = before - len(by_split["val"])

        val_g = gids(by_split["val"])

        before = len(by_split["train"])
        by_split["train"] = [
            r
            for r in by_split["train"]
            if (r["group_id"] not in test_g) and (r["group_id"] not in val_g)
        ]
        dropped_train = before - len(by_split["train"])

        if dropped_val or dropped_train:
            print(
                f"[group_disjoint] dropped from val: {dropped_val}, from train: {dropped_train}"
            )

        return by_split["train"] + by_split["val"] + by_split["test"]

    def _cap_rows(self, rows: list[dict], max_per_split: Optional[int]) -> list[dict]:
        if not max_per_split or max_per_split <= 0:
            return rows
        by_split = {"train": [], "val": [], "test": []}
        for r in rows:
            s = r["split"]
            if s in by_split:
                if len(by_split[s]) < max_per_split:
                    by_split[s].append(r)
        return by_split["train"] + by_split["val"] + by_split["test"]

    def write(
        self, manifest_path: Path, *, max_per_split: Optional[int] = None
    ) -> Path:
        out_path = Path(manifest_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        all_rows = self._dedupe_rows(self._rows)
        all_rows = self._enforce_group_disjoint(all_rows)
        all_rows = self._cap_rows(all_rows, max_per_split)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["path", "label", "split", "source", "group_id"]
            )
            w.writeheader()
            w.writerows(all_rows)

        return out_path


if __name__ == "__main__":
    if any((not os.path.exists(missing := p)) for p in [ROOT, RAW_DATA_DIR, FoR_DIR]):
        raise FileNotFoundError(
            f"Missing file or directory: {missing}. Please refer to {ROOT / 'README.md'} for details"
        )
    manifest_fp = ROOT / "manifest.csv"

    builder = ManifestBuilder()

    builder.add_rows_from_patterns(
        "FoR-norm",
        training_pattern="./data/raw/for-norm/training/**/*.wav",
        testing_pattern="./data/raw/for-norm/testing/**/*.wav",
        validation_pattern="./data/raw/for-norm/validation/**/*.wav",
        group_id_from_name=builder.group_id_from_filename_for,
        determine_cls_from_filepath=builder.determine_cls_from_filepath_for,
    )

    if CODECFAKE_PLUS_CORS_DIR.exists() and CODECFAKE_PLUS_CORS_LABELS.exists():
        builder.add_rows_from_codecfake_labels(
            CODECFAKE_PLUS_CORS_DIR,
            CODECFAKE_PLUS_CORS_LABELS,
            split_ratio=CODECFAKE_SPLIT_RATIO,
            source_name="CodecFakePlus-CoRS",
            group_id_from_name=builder._group_id_from_codecfake_cors_name,
        )

    else:
        print("Missing")
    if CODECFAKE_PLUS_COSG_DIR.exists() and CODECFAKE_PLUS_COSG_LABELS.exists():
        builder.add_rows_from_codecfake_labels(
            CODECFAKE_PLUS_COSG_DIR,
            CODECFAKE_PLUS_COSG_LABELS,
            split_ratio=CODECFAKE_SPLIT_RATIO,
            source_name="CodecFakePlus-CoSG",
            group_id_from_name=builder._group_id_from_codecfake_cosg_name,
        )

    builder.write(manifest_fp)
    print("Manifest has been built at:", manifest_fp)
