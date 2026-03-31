"""BERTScore computation for R₀, R₁, R₂ against reference answers."""

from typing import List, Optional

from bert_score import score as bert_score


def compute_bertscore(candidates: List[str], references: List[str]) -> List[dict]:
    """Compute BERTScore P, R, F1 for (candidate, reference) pairs.

    Uses roberta-large backbone, pinned via bert-score==0.3.13.
    """
    P, R, F1 = bert_score(
        candidates,
        references,
        model_type="roberta-large",
        lang="en",
        verbose=False,
        device="cpu",
    )
    return [
        {"p": round(p, 4), "r": round(r, 4), "f1": round(f, 4)}
        for p, r, f in zip(P.tolist(), R.tolist(), F1.tolist())
    ]


def score_all(
    r0_list: List[Optional[str]],
    r1_list: List[Optional[str]],
    r2_text_list: List[Optional[str]],
    reference_list: List[str],
) -> List[dict]:
    """Score R₀, R₁, R₂ for all queries in batched calls per layer.

    Returns list of dicts with p/r/f1 for each layer, plus F1 and Recall deltas.
    Handles None values (failed queries) by returning None for those score fields.
    """
    empty = {
        "p_r0": None, "r_r0": None, "f1_r0": None,
        "p_r1": None, "r_r1": None, "f1_r1": None,
        "p_r2": None, "r_r2": None, "f1_r2": None,
        "delta1_f1": None, "delta2_f1": None,
        "delta1_recall": None, "delta2_recall": None,
    }
    results = [dict(empty) for _ in reference_list]

    for prefix, candidates in [("r0", r0_list), ("r1", r1_list), ("r2", r2_text_list)]:
        valid_indices = [i for i, c in enumerate(candidates) if c is not None]
        if not valid_indices:
            continue
        valid_candidates = [candidates[i] for i in valid_indices]
        valid_references = [reference_list[i] for i in valid_indices]
        scores = compute_bertscore(valid_candidates, valid_references)
        for idx, s in zip(valid_indices, scores):
            results[idx][f"p_{prefix}"] = s["p"]
            results[idx][f"r_{prefix}"] = s["r"]
            results[idx][f"f1_{prefix}"] = s["f1"]

    for r in results:
        if r["f1_r0"] is not None and r["f1_r1"] is not None:
            r["delta1_f1"] = round(r["f1_r1"] - r["f1_r0"], 4)
            r["delta1_recall"] = round(r["r_r1"] - r["r_r0"], 4)
        if r["f1_r1"] is not None and r["f1_r2"] is not None:
            r["delta2_f1"] = round(r["f1_r2"] - r["f1_r1"], 4)
            r["delta2_recall"] = round(r["r_r2"] - r["r_r1"], 4)

    return results
