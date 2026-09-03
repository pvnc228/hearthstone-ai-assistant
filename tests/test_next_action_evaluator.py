from src.llm.evaluate_next_action import score_response, summarize_scores


def record(chosen=2):
    return {
        "chosen_candidate_id": chosen,
        "candidates": [
            {"id": 1, "type": "END_TURN"},
            {"id": 2, "type": "PLAY"},
        ],
    }


def test_evaluator_separates_format_existence_and_accuracy():
    scores = [
        score_response(record(), "PLAN: [2]", latency_ms=10),
        score_response(record(), "PLAN: [9]", latency_ms=20),
        score_response(record(), "free text", latency_ms=30),
    ]
    metrics = summarize_scores(scores)

    assert metrics["total"] == 3
    assert metrics["top1_accuracy"] == 1 / 3
    assert metrics["format_valid_rate"] == 2 / 3
    assert metrics["candidate_exists_rate"] == 1 / 3
    assert metrics["by_action_type"]["PLAY"]["total"] == 3
    assert metrics["latency_ms"]["p50"] == 20
