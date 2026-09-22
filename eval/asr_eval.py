"""CLI: WER/CER report on the downloaded FLEURS subset.

SPEC.md Section 11.3 / M2 accept criterion: "per-language WER report on a
FLEURS subset (reported, no threshold yet)". Requires
`uv run python scripts/download_datasets.py --only fleurs` first.

Usage: uv run python -m eval.asr_eval [--languages en es hi tl]
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel

from eval.metrics.asr import compute_wer_by_language

REPO_ROOT = Path(__file__).resolve().parent.parent
FLEURS_CACHE = REPO_ROOT / "data" / "cache" / "fleurs"


def load_manifest(lang: str) -> list[dict]:
    manifest_path = FLEURS_CACHE / lang / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} missing — run "
            "'uv run python scripts/download_datasets.py --only fleurs' first"
        )
    with manifest_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", default=["en", "es", "hi", "tl"])
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--compute-type", default="int8")
    args = parser.parse_args()

    model = WhisperModel(args.model_size, device="cpu", compute_type=args.compute_type)

    by_language: dict[str, list[tuple[str, str]]] = {}
    for lang in args.languages:
        pairs = []
        for entry in load_manifest(lang):
            data, _sample_rate = sf.read(REPO_ROOT / entry["path"])
            segments, _info = model.transcribe(data.astype("float32"), language=lang)
            hypothesis = " ".join(segment.text.strip() for segment in segments)
            pairs.append((entry["transcription"], hypothesis))
        by_language[lang] = pairs

    results = sorted(compute_wer_by_language(by_language), key=lambda r: r.lang)

    run_id = f"asr-wer-{int(time.time())}"
    run_dir = REPO_ROOT / "reports" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# ASR WER/CER report (FLEURS subset)",
        "",
        f"Model: faster-whisper {args.model_size}, compute_type={args.compute_type}, "
        "device=cpu, git commit: (see reports/<run_id>)",
        "",
        "| Language | Clips | WER | CER |",
        "|---|---|---|---|",
    ]
    for result in results:
        lines.append(f"| {result.lang} | {result.n_clips} | {result.wer:.3f} | {result.cer:.3f} |")

    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )

    print("\n".join(lines))
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
