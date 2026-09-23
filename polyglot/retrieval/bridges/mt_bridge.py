"""MT bridge: translate the query to English with NLLB, embed with BGE-M3,
dense search. See SPEC.md Section 8.7.

Verified against installed transformers==5.17.0: the generic pipeline(
"translation", ...) task was removed from this version's PIPELINE_REGISTRY
(KeyError: Unknown task translation). Uses AutoModelForSeq2SeqLM.generate()
directly instead. FLORES-200 language codes confirmed present in the
installed facebook/nllb-200-distilled-600M tokenizer's vocab: eng_Latn,
spa_Latn, hin_Deva, tgl_Latn (NLLB uses "tgl_Latn" for Tagalog; note this
differs from FLEURS' "fil_ph" config name for the same language family —
different datasets, different naming, both meaning Tagalog/Filipino).
"""

import asyncio

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from polyglot.core.types import Passage
from polyglot.retrieval.store import RetrievalStore

NLLB_LANG_CODES = {"en": "eng_Latn", "es": "spa_Latn", "hi": "hin_Deva", "tl": "tgl_Latn"}


class Translator:
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M") -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def translate_to_english(self, text: str, src_lang: str) -> str:
        if src_lang == "en":
            return text
        src_code = NLLB_LANG_CODES.get(src_lang)
        if src_code is None:
            raise ValueError(f"no NLLB language code configured for {src_lang!r}")

        inputs = self._tokenizer(text, src_lang=src_code, return_tensors="pt")
        target_id = self._tokenizer.convert_tokens_to_ids(NLLB_LANG_CODES["en"])
        output_ids = self._model.generate(
            **inputs, forced_bos_token_id=target_id, max_new_tokens=200
        )
        return self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]


class MtBridge:
    def __init__(
        self, store: RetrievalStore, embedder: SentenceTransformer, translator: Translator
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._translator = translator

    async def retrieve(self, query: str, lang: str, k: int) -> list[Passage]:
        loop = asyncio.get_running_loop()
        english_query = await loop.run_in_executor(
            None, self._translator.translate_to_english, query, lang
        )
        embedding = await loop.run_in_executor(None, self._embed, english_query)
        return self._store.search(embedding, k)

    def _embed(self, text: str) -> np.ndarray:
        return self._embedder.encode(text, normalize_embeddings=True)
