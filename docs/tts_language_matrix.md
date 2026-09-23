# TTS language matrix

Built in M4. Every number here came from an actual script run on the
reference dev hardware (CLAUDE.md rule 4) — see the commands under each
section to reproduce. No GPU has been used for TTS yet; all measurements
are CPU-only.

**Hardware:** AMD Ryzen 7 7800X3D (8 cores / 16 threads), CPU-only, Windows.
No GPU tested for TTS as of M4 (the M0 GPU/Modal decision was never acted
on).

**The fourth project language was swapped from Tagalog to Mandarin
mid-M4**, after this matrix's first pass found that none of SPEC.md's three
named TTS engines (CosyVoice2, Kokoro, Piper) support Tagalog at all —
checked directly against each project's published voice/language list, not
assumed. Mandarin is covered by all three (it's CosyVoice2's *primary*
language). The Tagalog findings are kept below for the record, since they're
the reason for the swap.

## Summary

| Engine | Languages covered (of en/es/hi/zh) | License | CPU-viable | Notes |
|---|---|---|---|---|
| **Kokoro-82M** | en, es, hi, zh | Apache 2.0 | Yes — used as the M4 default | TTFB 0.77-1.0s on this CPU (see below) — a real bottleneck against the <800ms perceived-latency target. |
| **Piper** | en, es, hi | MIT (voices vary; see below) | Yes | No Mandarin voice in `rhasspy/piper-voices`. Not wired into `polyglot/tts/` — Kokoro already covers en/es/hi and, unlike Piper, also covers zh. |
| **CosyVoice2-0.5B** | en, zh (not es, hi) | Apache 2.0 | Needs verification — not evaluated | Mandarin is this model's primary language, but it still doesn't cover Spanish or Hindi, so it can't be the sole engine regardless. |
| **XTTS-v2** | — | Non-commercial | N/A | Explicitly excluded — CLAUDE.md rule: "Do not use XTTS-v2 (non-commercial license)." Listed only for completeness. |

## Kokoro-82M (chosen for M4, all 4 languages)

`polyglot/tts/kokoro_engine.py`. Verified against installed `kokoro==0.9.4`:

| Language | `lang_code` | Default voice used | TTFB (this CPU) | Notes |
|---|---|---|---|---|
| en | `a` | `af_heart` | 765ms | |
| es | `e` | `ef_dora` | 782ms | |
| hi | `h` | `hf_alpha` | 1000ms | |
| zh | `z` | `zf_xiaobei` | 843ms | Needs the `misaki[zh]` extras (`cn2an`, `jieba`, `ordered-set`, `pypinyin`, `pypinyin-dict`) — now in `pyproject.toml`. `jieba` builds a word-segmentation dictionary on first use (~0.4s, one-time per process, excluded from the TTFB figure below). |

TTFB measured as wall-clock time from calling `pipeline(text, voice=...)`
to the first yielded result, for one representative airline-refund
sentence per language, CPU, cold pipeline already loaded and (for Mandarin)
`jieba`'s dictionary already warmed up — that's a one-time startup cost,
not per-turn latency. Reproduce:

```
uv run python -c "
import time
from kokoro import KPipeline
for lang_code, text, voice in [
    ('a', 'Your flight has been cancelled and you are entitled to a full refund.', 'af_heart'),
    ('e', 'Su vuelo ha sido cancelado y tiene derecho a un reembolso completo.', 'ef_dora'),
    ('h', 'आपकी उड़ान रद्द कर दी गई है और आप पूर्ण धनवापसी के हकदार हैं।', 'hf_alpha'),
    ('z', '您的航班已取消，您有权获得全额退款。', 'zf_xiaobei'),
]:
    pipeline = KPipeline(lang_code=lang_code)
    for _ in pipeline('warmup', voice=voice): break  # excludes jieba dict build for zh
    start = time.monotonic()
    for result in pipeline(text, voice=voice):
        print(lang_code, 'TTFB ms:', round((time.monotonic()-start)*1000)); break
"
```

**These TTFB numbers alone (0.77-1.0s) exceed SPEC.md Section 3's p50
perceived-latency target of 500ms**, before VAD/ASR/retrieval/LLM time is
even added. On CPU, Kokoro is a real bottleneck, not a minor one, across
all four languages. Whether this resolves on a GPU is unverified — no GPU
has been tested for TTS.

No espeak-ng install was needed for es/hi in this testing, even though
`VOICES.md` lists it as those languages' fallback phonemizer — misaki's
bundled G2P produced audio without it here. Not something to assume holds
on a different machine; if quality looks off on real hardware, check
whether `misaki` silently degraded without espeak-ng available.

License: Apache 2.0 (model), individual voice licenses not separately
checked per-voice.

## Piper (documented, not wired in)

Confirmed via `rhasspy/piper-voices` on HuggingFace: voice directories
exist for `en`, `es`, `hi`; none for Mandarin (`zh`/`cmn`) either, as it
happens. Not installed or benchmarked in M4 — Kokoro already covers all
four languages and was built first. Worth reconsidering if Kokoro's CPU
TTFB turns out to be a hard blocker and a GPU isn't available, since
Piper's ONNX runtime models are typically much faster per-utterance than
Kokoro's.

License: MIT for the Piper project itself; individual voices are licensed
separately (mostly CC0/CC-BY, some MIT) — check the specific voice's card
before shipping if this gets adopted.

## CosyVoice2-0.5B (named in SPEC.md, only usable for 2 of 4 languages)

Per the model's own published language list: Chinese, English, Japanese,
Korean, and Chinese dialects (Cantonese, Sichuanese, Shanghainese, etc.).
Covers en and zh but not es or hi, so it can't be the project's sole
engine even after the language swap — noted here because SPEC.md Section 5
names it first, and it's worth recording that its Mandarin coverage is
real (Mandarin is its primary language) in case Kokoro's CPU latency
becomes a hard blocker and this is worth benchmarking properly.

## Historical: why Tagalog was dropped

The original fourth-language choice (SPEC.md M0 open question, decided as
Tagalog) turned out to have **no support in any of the three named TTS
engines**:

| Language | CosyVoice2 | Kokoro | Piper |
|---|---|---|---|
| Tagalog (`tl`/`fil`) | Not supported | No `lang_code` exists | No `tl`/`fil` directory in `rhasspy/piper-voices` |

This wasn't a CPU-vs-GPU limitation — all three gaps are fixed by each
project's training data / voice catalog, not compute. Options considered:
accept text-only responses for Tagalog callers, evaluate Meta's MMS-TTS
(claims 1100+ languages, never checked), or swap the language. The human
chose to swap to Mandarin, which resolved the gap immediately (see Summary
table above).

## Not evaluated

- **GPU inference for any of the above** — no GPU has been set up (M0's
  Modal/GPU decision is still open). All TTFB numbers above are CPU-only
  and should be re-measured once GPU access exists.
- **Meta MMS-TTS** — no longer needed for language coverage now that
  Mandarin is the fourth language, but could still be worth a look if
  Kokoro's CPU latency doesn't improve on GPU.
