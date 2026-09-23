"""CLI: recall@k / MRR report per language per bridge, against the real
golden set.

SPEC.md Section 11.3 / M3 accept criterion. BLOCKED until:
1. config/sources.yaml has real KB URLs (human-provided — CLAUDE.md rule 6,
   Claude Code does not invent them) and `python -m polyglot.retrieval.ingest`
   has been run against them.
2. eval/golden/golden.jsonl has real, human-reviewed items whose
   expected_passage_ids match real ingested chunk ids.

Refuses to run rather than fabricate numbers (CLAUDE.md rule 4) — raises a
clear error naming what's missing instead.

Usage: uv run python -m eval.retrieval_eval [--k 10]
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer

from eval.golden.schema import GoldenItem
from eval.metrics.retrieval import RetrievalResult, summarize
from polyglot.retrieval.bridges.hybrid import HybridBridge
from polyglot.retrieval.bridges.mt_bridge import MtBridge, Translator
from polyglot.retrieval.bridges.multilingual import MultilingualBridge
from polyglot.retrieval.ingest import DEFAULT_DSN
from polyglot.retrieval.store import RetrievalStore

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = REPO_ROOT / "eval" / "golden" / "golden.jsonl"


def load_golden() -> list[GoldenItem]:
    if not GOLDEN_PATH.exists():
        raise SystemExit(
            f"{GOLDEN_PATH} does not exist yet. SPEC.md Section 10.3 wants at least "
            "100 human-reviewed items, built after config/sources.yaml is filled in "
            "and the corpus is ingested. Nothing to evaluate yet."
        )
    items = []
    with GOLDEN_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(GoldenItem.model_validate(json.loads(line)))
    return items


async def evaluate_bridge(
    bridge_name: str,
    bridge: MtBridge | MultilingualBridge | HybridBridge,
    items: list[GoldenItem],
    k: int,
) -> list[RetrievalResult]:
    results = []
    for item in items:
        passages = await bridge.retrieve(item.reference_transcript, item.lang, k)
        results.append(
            RetrievalResult(
                query_id=item.id,
                lang=item.lang,
                bridge=bridge_name,
                expected_passage_ids=set(item.expected_passage_ids),
                retrieved_passage_ids=[p.id for p in passages],
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    items = load_golden()

    dsn = os.environ.get("POLYGLOT_DB_DSN", DEFAULT_DSN)
    store = RetrievalStore(dsn)
    embedder = SentenceTransformer("BAAI/bge-m3")
    translator = Translator()
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    mt = MtBridge(store, embedder, translator)
    multilingual = MultilingualBridge(store, embedder)
    hybrid = HybridBridge(mt, multilingual, reranker=reranker)

    all_results: list[RetrievalResult] = []
    for name, bridge in [("mt", mt), ("multilingual", multilingual), ("hybrid", hybrid)]:
        all_results.extend(asyncio.run(evaluate_bridge(name, bridge, items, args.k)))

    reports = summarize(all_results)

    run_id = f"retrieval-{int(time.time())}"
    run_dir = REPO_ROOT / "reports" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Retrieval recall@k / MRR report",
        "",
        "| Language | Bridge | Queries | Recall@1 | Recall@5 | Recall@10 | MRR |",
        "|---|---|---|---|---|---|---|",
    ]
    for report in reports:
        lines.append(
            f"| {report.lang} | {report.bridge} | {report.n_queries} | "
            f"{report.recall_at_1:.3f} | {report.recall_at_5:.3f} | "
            f"{report.recall_at_10:.3f} | {report.mrr:.3f} |"
        )

    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps([asdict(r) for r in reports], indent=2), encoding="utf-8"
    )

    print("\n".join(lines))
    print(f"\nwrote {run_dir}")
    store.close()


if __name__ == "__main__":
    main()
