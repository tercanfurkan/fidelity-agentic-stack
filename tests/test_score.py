"""Tests for eval/score.py — validates scoring logic and delta computation."""

from eval.score import compute_bertscore, score_all


def test_compute_bertscore_returns_correct_structure():
    candidates = ["The cat sat on the mat."]
    references = ["The cat sat on the mat."]
    results = compute_bertscore(candidates, references)
    assert len(results) == 1
    assert set(results[0].keys()) == {"p", "r", "f1"}
    assert 0.0 <= results[0]["f1"] <= 1.0
    assert 0.0 <= results[0]["p"] <= 1.0
    assert 0.0 <= results[0]["r"] <= 1.0


def test_compute_bertscore_identical_texts_high_score():
    text = "httpx raises TimeoutException on timeout."
    results = compute_bertscore([text], [text])
    assert results[0]["f1"] > 0.99


def test_score_all_handles_none_values():
    r0_list = ["some text", None]
    r1_list = ["some answer", "another answer"]
    r2_list = ["formatted text", "formatted answer"]
    references = ["reference one", "reference two"]
    results = score_all(r0_list, r1_list, r2_list, references)
    assert len(results) == 2
    assert results[0]["f1_r0"] is not None
    assert results[0]["delta1_f1"] is not None
    assert results[1]["f1_r0"] is None
    assert results[1]["f1_r1"] is not None
    assert results[1]["delta1_f1"] is None  # can't compute delta if r0 is None


def test_score_all_computes_deltas():
    r0 = ["Pass a timeout parameter to the request method or Client constructor."]
    r1 = ["Use httpx.get(url, timeout=5.0) to set a timeout."]
    r2 = ["Set timeout with httpx.get timeout parameter."]
    ref = ["Pass a timeout parameter to the request method or Client constructor. httpx.get(url, timeout=5.0) sets a 5-second timeout."]
    results = score_all(r0, r1, r2, ref)
    r = results[0]
    assert r["delta1_f1"] is not None
    assert r["delta2_f1"] is not None
    assert abs(r["delta1_f1"] - (r["f1_r1"] - r["f1_r0"])) < 0.0001
    assert abs(r["delta2_f1"] - (r["f1_r2"] - r["f1_r1"])) < 0.0001
    assert abs(r["delta1_recall"] - (r["r_r1"] - r["r_r0"])) < 0.0001
    assert abs(r["delta2_recall"] - (r["r_r2"] - r["r_r1"])) < 0.0001
