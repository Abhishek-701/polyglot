# TTS language matrix

Built in M4. Every number here came from an actual script run on the
reference dev hardware (CLAUDE.md rule 4) — see the commands under each
section to reproduce. No GPU has been used for TTS yet; all measurements
are CPU-only.

**Hardware:** AMD Ryzen 7 7800X3D (8 cores / 16 threads), CPU-only, Windows.
No GPU tested for TTS as of M4 (the M0 GPU/Modal decision was never acted
on).

## Summary

| Engine | Languages covered (of en/es/hi/tl) | License | CPU-viable | Notes |
|---|---|---|---|---|
| **Kokoro-82M** | en, es, hi | Apache 2.0 | Yes — used as the M4 default | No Tagalog. TTFB 0.8-1.0s on this CPU (see below) — a real bottleneck against the <800ms perceived-latency target. |
| **Piper** | en, es, hi | MIT (voices vary; see below) | Yes | Not wired into `polyglot/tts/` yet — Kokoro covers the same 3 languages and was already built out. Kept as a documented CPU fallback option per SPEC.md Section 5. |
| **CosyVoice2-0.5B** | none of ours (Chinese, English, Japanese, Korean + Chinese dialects) | Apache 2.0 | Needs verification — not evaluated | SPEC.md names this as the primary engine, but it doesn't support Spanish, Hindi, or Tagalog at all. Not usable for 3 of our 4 languages regardless of hardware. |
| **XTTS-v2** | — | Non-commercial | N/A | Explicitly excluded — CLAUDE.md rule: "Do not use XTTS-v2 (non-commercial license)." Listed only for completeness. |

**The real finding: none of SPEC.md's three named engines (CosyVoice2,
Kokoro, Piper) support Tagalog.** This isn't a CPU-vs-GPU limitation —
CosyVoice2 and Kokoro's language lists are fixed by training data, and
Piper's voice catalog (`rhasspy/piper-voices` on HuggingFace) has no
`tl`/`fil` directory. Checked directly against each project's published
voice/language list, not assumed. Open question for the human: accept no
native TTS for the fourth language (text-only fallback for Tagalog
responses), evaluate Meta's MMS-TTS (claims 1100+ languages, not verified
here), or reconsider the fourth-language choice. Recorded in PROGRESS.md.

## Kokoro-82M (chosen for M4)

`polyglot/tts/kokoro_engine.py`. Verified against installed `kokoro==0.9.4`:

| Language | `lang_code` | Default voice used | TTFB (this CPU) | Notes |
|---|---|---|---|---|
| en | `a` | `af_heart` | 765ms | |
| es | `e` | `ef_dora` | 782ms | |
| hi | `h` | `hf_alpha` | 1000ms | |
| tl | — | — | — | Not supported. No `lang_code` exists for Tagalog/Filipino. |

TTFB measured as wall-clock time from calling `pipeline(text, voice=...)`
to the first yielded result, for one representative airline-refund
sentence per language, CPU, cold pipeline already loaded (model load time
excluded — that's a one-time startup cost, not per-turn latency).
Reproduce:

```
uv run python -c "
import time
from kokoro import KPipeline
for lang_code, text, voice in [
    ('a', 'Your flight has been cancelled and you are entitled to a full refund.', 'af_heart'),
    ('e', 'Su vuelo ha sido cancelado y tiene derecho a un reembolso completo.', 'ef_dora'),
    ('h', 'आपकी उड़ान रद्द कर दी गई है और आप पूर्ण धनवापसी के हकदार हैं।', 'hf_alpha'),
]:
    pipeline = KPipeline(lang_code=lang_code)
    start = time.monotonic()
    for result in pipeline(text, voice=voice):
        print(lang_code, 'TTFB ms:', round((time.monotonic()-start)*1000)); break
"
```

**These TTFB numbers alone (0.8-1.0s) exceed SPEC.md Section 3's p50
perceived-latency target of 500ms**, before VAD/ASR/retrieval/LLM time is
even added. On CPU, Kokoro is a real bottleneck, not a minor one. Whether
this resolves on a GPU is unverified — no GPU has been tested for TTS.

No espeak-ng install was needed for es/hi in this testing, even though
`VOICES.md` lists it as those languages' fallback phonemizer — misaki's
bundled G2P produced audio without it here. Not something to assume holds
on a different machine; if quality looks off on real hardware, check
whether `misaki` silently degraded without espeak-ng available.

License: Apache 2.0 (model), individual voice licenses not separately
checked per-voice.

## Piper (documented, not wired in)

Confirmed via `rhasspy/piper-voices` on HuggingFace: voice directories
exist for `en`, `es`, `hi`; none for `tl`/`fil`. Not installed or
benchmarked in M4 — Kokoro already covers the same three languages and was
built first. Worth reconsidering if Kokoro's CPU TTFB turns out to be a
hard blocker and a GPU isn't available, since Piper's ONNX runtime models
are typically much faster per-utterance than Kokoro's.

License: MIT for the Piper project itself; individual voices are licensed
separately (mostly CC0/CC-BY, some MIT) — check the specific voice's card
before shipping if this gets adopted.

## CosyVoice2-0.5B (named in SPEC.md, not usable for our languages)

Per the model's own published language list: Chinese, English, Japanese,
Korean, and Chinese dialects (Cantonese, Sichuanese, Shanghainese, etc.).
No Spanish, Hindi, or Tagalog. Only 1 of our 4 target languages (English)
is covered at all, so this isn't a realistic primary engine for this
project regardless of CPU/GPU — noted here because SPEC.md Section 5 names
it first, and it's important to record why it wasn't pursued rather than
silently dropping it.

## Not evaluated

- **Meta MMS-TTS** — claims very broad language coverage (1000+
  languages) and is a plausible candidate for Tagalog specifically, but
  wasn't checked in M4. Worth evaluating before accepting "no Tagalog TTS"
  as final.
- **GPU inference for any of the above** — no GPU has been set up (M0's
  Modal/GPU decision is still open). All TTFB numbers above are CPU-only
  and should be re-measured once GPU access exists.
