"""Downloads the M2 eval datasets. Never commit their output (CLAUDE.md data rules).

Writes everything under data/cache/ (gitignored):
- data/cache/models/lid.176.ftz         fastText LID model (CC BY-SA 3.0)
- data/cache/fleurs/<lang>/*.wav + manifest.jsonl   FLEURS subset (CC BY 4.0)
- data/cache/noise/esc10/*.wav           ESC-10 subset of ESC-50 (CC BY 3.0)
- data/cache/silence/*.wav               generated (no license needed)

See docs/data_licenses.md for the full license record (SPEC.md/CLAUDE.md
data rules require this for every dataset and model used).

Usage: uv run python scripts/download_datasets.py [--only lid|fleurs|noise|silence]
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "cache"

LID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"

# google/fleurs config name per language (SPEC.md fourth language: Tagalog / fil_ph).
FLEURS_CONFIGS = {"en": "en_us", "es": "es_419", "hi": "hi_in", "tl": "fil_ph"}

ESC50_META_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
ESC50_AUDIO_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{filename}"


def download_lid_model() -> Path:
    dest = CACHE_DIR / "models" / "lid.176.ftz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"lid model already present: {dest}")
        return dest
    response = requests.get(LID_MODEL_URL, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    print(f"wrote {dest} ({len(response.content)} bytes)")
    return dest


def download_fleurs_subset(languages: list[str], n_per_language: int = 10) -> None:
    from datasets import Audio, load_dataset

    for lang in languages:
        config = FLEURS_CONFIGS[lang]
        lang_dir = CACHE_DIR / "fleurs" / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = lang_dir / "manifest.jsonl"
        if manifest_path.exists():
            print(f"fleurs/{lang} already present: {manifest_path}")
            continue

        ds = load_dataset("google/fleurs", config, split="test", streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))

        entries = []
        for i, example in enumerate(ds):
            if i >= n_per_language:
                break
            wav_path = lang_dir / f"{example['id']}.wav"
            data, sample_rate = sf.read(io.BytesIO(example["audio"]["bytes"]))
            sf.write(wav_path, data, sample_rate)
            entries.append(
                {
                    "id": example["id"],
                    "lang": lang,
                    "path": str(wav_path.relative_to(REPO_ROOT)),
                    "sample_rate": sample_rate,
                    "transcription": example["transcription"],
                    "raw_transcription": example["raw_transcription"],
                }
            )

        with manifest_path.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        print(f"wrote {len(entries)} clips to {lang_dir}")


def download_esc10_noise(n_per_category: int = 2) -> None:
    dest_dir = CACHE_DIR / "noise" / "esc10"
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_dir / "manifest.jsonl"
    if manifest_path.exists():
        print(f"esc10 noise already present: {manifest_path}")
        return

    meta_response = requests.get(ESC50_META_URL, timeout=30)
    meta_response.raise_for_status()
    lines = meta_response.text.splitlines()
    header = lines[0].split(",")
    rows = [dict(zip(header, line.split(","), strict=True)) for line in lines[1:]]
    esc10_rows = [r for r in rows if r["esc10"] == "True"]

    by_category: dict[str, list[dict[str, str]]] = {}
    for row in esc10_rows:
        by_category.setdefault(row["category"], []).append(row)

    entries = []
    for category, category_rows in sorted(by_category.items()):
        for row in category_rows[:n_per_category]:
            filename = row["filename"]
            audio_response = requests.get(ESC50_AUDIO_URL.format(filename=filename), timeout=30)
            audio_response.raise_for_status()
            wav_path = dest_dir / filename
            wav_path.write_bytes(audio_response.content)
            entries.append({"path": str(wav_path.relative_to(REPO_ROOT)), "category": category})

    with manifest_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    print(f"wrote {len(entries)} noise clips to {dest_dir}")


def generate_silence_clips(n: int = 5, duration_s: float = 3.0, sample_rate: int = 16000) -> None:
    dest_dir = CACHE_DIR / "silence"
    dest_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=0)
    n_samples = int(duration_s * sample_rate)

    for i in range(n):
        # Barely-there dither (-60dBFS) instead of exact digital zero, closer
        # to a real quiet room than a degenerate all-zero input.
        noise_floor = rng.normal(0, 0.001, n_samples).astype(np.float32)
        path = dest_dir / f"silence-{i:02d}.wav"
        sf.write(path, noise_floor, sample_rate)
    print(f"wrote {n} silence clips to {dest_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", choices=["lid", "fleurs", "noise", "silence"], action="append", default=None
    )
    parser.add_argument("--languages", nargs="+", default=list(FLEURS_CONFIGS.keys()))
    parser.add_argument("--fleurs-n", type=int, default=10)
    args = parser.parse_args()

    steps = args.only or ["lid", "fleurs", "noise", "silence"]

    if "lid" in steps:
        download_lid_model()
    if "fleurs" in steps:
        download_fleurs_subset(args.languages, n_per_language=args.fleurs_n)
    if "noise" in steps:
        download_esc10_noise()
    if "silence" in steps:
        generate_silence_clips()


if __name__ == "__main__":
    main()
