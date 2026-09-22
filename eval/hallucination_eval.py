"""CLI: hallucination guard report on silence and noise clips.

SPEC.md Section 11.3 / M2 accept criterion: "hallucination report on silence
and noise clips". Requires
`uv run python scripts/download_datasets.py --only silence --only noise` first.

Usage: uv run python -m eval.hallucination_eval
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel

from eval.metrics.hallucination import ClipResult, summarize
from polyglot.asr.hallucination_guard import HallucinationGuardConfig, check_hallucination

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_CACHE = REPO_ROOT / "data" / "cache"


def _load_clips(category: str) -> list[Path]:
    directory = DATA_CACHE / "silence" if category == "silence" else DATA_CACHE / "noise" / "esc10"
    if not directory.exists():
        raise FileNotFoundError(
            f"{directory} missing — run "
            f"'uv run python scripts/download_datasets.py --only {category}' first"
        )
    return sorted(directory.glob("*.wav"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--compute-type", default="int8")
    args = parser.parse_args()

    model = WhisperModel(args.model_size, device="cpu", compute_type=args.compute_type)
    guard_config = HallucinationGuardConfig()

    results: list[ClipResult] = []
    for category in ("silence", "noise"):
        for path in _load_clips(category):
            data, _sample_rate = sf.read(path)
            segments, _info = model.transcribe(data.astype("float32"))
            kept_texts = []
            reason: str | None = None
            for segment in segments:
                segment_reason = check_hallucination(
                    segment.text,
                    segment.no_speech_prob,
                    segment.avg_logprob,
                    segment.compression_ratio,
                    guard_config,
                )
                if segment_reason is None:
                    kept_texts.append(segment.text.strip())
                else:
                    reason = reason or segment_reason
            results.append(
                ClipResult(
                    path=str(path.relative_to(REPO_ROOT)),
                    category=category,
                    kept_text=" ".join(kept_texts).strip(),
                    suppressed_reason=reason,
                )
            )

    reports = summarize(results)

    run_id = f"hallucination-{int(time.time())}"
    run_dir = REPO_ROOT / "reports" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Hallucination guard report",
        "",
        f"Model: faster-whisper {args.model_size}, compute_type={args.compute_type}, device=cpu",
        "",
        "| Category | Clips | Non-empty rate (post-guard) | Suppressions by reason |",
        "|---|---|---|---|",
    ]
    for report in reports:
        lines.append(
            f"| {report.category} | {report.n_clips} | {report.non_empty_rate:.1%} | "
            f"{report.suppressions_by_reason} |"
        )

    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps([asdict(r) for r in reports], indent=2), encoding="utf-8"
    )
    (run_dir / "clips.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n".join(lines))
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
