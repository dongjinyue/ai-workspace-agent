import pytest

from tests.evals.runner import evaluate, load_cases, production_path_predictor


pytestmark = pytest.mark.eval


def test_golden_dataset_routing_and_no_hit_safety(capsys):
    result = evaluate(production_path_predictor, load_cases())
    print("\n" + result.summary())

    assert result.tool_correct == result.tool_total
    assert result.skill_correct == result.skill_total
    assert result.no_hit_correct == result.no_hit_total
    assert result.retrieval_correct == result.retrieval_total
    assert result.security_correct == result.security_total

    output = capsys.readouterr().out
    assert "Evaluation Summary" in output
