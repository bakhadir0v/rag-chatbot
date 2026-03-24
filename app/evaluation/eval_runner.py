"""
MLOps evaluation loop.

Usage:
    python -m app.evaluation.eval_runner \
        --dataset data/eval_dataset.json \
        --variants default verbose \
        --output data/eval_results.json
"""
import argparse
import json
import logging
import statistics
import time
from pathlib import Path

from app.rag.chain import ask
from app.rag.retriever import retrieve
from app.evaluation.metrics import retrieval_precision, answer_relevance_score

logger = logging.getLogger(__name__)


def load_dataset(path: str) -> list[dict]:
    """
    Expected format:
    [
      {
        "question": "...",
        "reference_answer": "...",
        "relevant_sources": ["doc1.pdf", "doc2.pdf"]
      },
      ...
    ]
    """
    with open(path) as f:
        return json.load(f)


def run_eval(dataset: list[dict], variants: list[str]) -> dict:
    results = {}

    for variant in variants:
        logger.info("Evaluating prompt variant: %s", variant)
        precision_scores, relevance_scores, latencies = [], [], []

        for item in dataset:
            question = item["question"]
            reference = item["reference_answer"]
            relevant_sources = item.get("relevant_sources", [])

            # Retrieval
            docs = retrieve(question)
            precision = retrieval_precision(docs, relevant_sources)

            # Generation
            t0 = time.perf_counter()
            result = ask(question, prompt_variant=variant)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)

            # Answer quality
            relevance = answer_relevance_score(question, reference, result["answer"])

            precision_scores.append(precision)
            relevance_scores.append(relevance)
            latencies.append(latency_ms)

            logger.info(
                "[%s] Q: %.50s | precision=%.2f | relevance=%.2f | latency=%.0fms",
                variant, question, precision, relevance, latency_ms,
            )

        results[variant] = {
            "retrieval_precision_mean": round(statistics.mean(precision_scores), 4),
            "answer_relevance_mean": round(statistics.mean(relevance_scores), 4),
            "latency_p50_ms": round(statistics.median(latencies), 2),
            "latency_p95_ms": round(
                sorted(latencies)[int(len(latencies) * 0.95)], 2
            ) if len(latencies) >= 2 else latencies[-1],
            "n_samples": len(dataset),
        }

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="RAG evaluation runner")
    parser.add_argument("--dataset", required=True, help="Path to eval dataset JSON")
    parser.add_argument(
        "--variants", nargs="+", default=["default"], help="Prompt variants to evaluate"
    )
    parser.add_argument("--output", default="data/eval_results.json", help="Output JSON path")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    logger.info("Loaded %d evaluation samples", len(dataset))

    results = run_eval(dataset, args.variants)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results written to %s", args.output)

    # Print summary table
    print("\n=== Evaluation Results ===")
    for variant, metrics in results.items():
        print(f"\nVariant: {variant}")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    # CI gate: fail if best variant is below thresholds
    best = max(results.values(), key=lambda m: m["answer_relevance_mean"])
    if best["retrieval_precision_mean"] < 0.6:
        raise SystemExit("FAILED: retrieval_precision below 0.6")
    if best["answer_relevance_mean"] < 0.7:
        raise SystemExit("FAILED: answer_relevance below 0.7")

    print("\nAll quality gates passed.")


if __name__ == "__main__":
    main()
