"""Integration checks execute real SDK metrics with an explicitly fake judge."""

import json
import socket

import pytest

import conversation_eval
from conversation_eval.__main__ import main
from conversation_eval.cases import make_case
from conversation_eval.replay import ReplayJudge
from conversation_eval.runner import evaluate_case, live_judge, make_metrics
from deepeval.metrics import ConversationalGEval, KnowledgeRetentionMetric, TurnRelevancyMetric
from deepeval.test_case import ConversationalTestCase


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline test attempted a network connection")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def test_native_case_and_metric_types(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_EVAL_MODE", "system_one")
    case = make_case()
    assert isinstance(case, ConversationalTestCase)
    assert len(case.turns) == 6
    metrics = make_metrics(ReplayJudge("retained"))
    assert [type(m) for m in metrics] == [TurnRelevancyMetric, KnowledgeRetentionMetric, ConversationalGEval]
    assert str(metrics[0].eval_mode) == str(metrics[1].eval_mode) == "llm"


@pytest.mark.parametrize("case, expected, passed", [
    ("retained", [1, 1, 0.9], True),
    ("forgotten", [2 / 3, 2 / 3, 0.3], False),
])
def test_actual_sdk_aggregation(case, expected, passed):
    report = evaluate_case(case)
    assert [result["score"] for result in report["results"]] == pytest.approx(expected)
    assert report["passed"] is passed
    assert report["mode"] == "scripted_replay"
    assert report["quality_evidence"] is False
    assert report["sdk_version"] == "4.3.1"
    assert all(result["reason"] for result in report["results"])
    assert report["replay_schema_calls"].count("Knowledge") == 3
    assert report["replay_schema_calls"].count("TurnRelevancyVerdict") == 3
    assert report["replay_schema_calls"].count("KnowledgeRetentionVerdict") == 3
    assert "ReasonScore" in report["replay_schema_calls"]


def test_unknown_case_rejected():
    with pytest.raises(ValueError, match="Unknown case"):
        make_case("arbitrary")


def test_unknown_response_schema_rejected():
    class Unsupported:
        pass
    with pytest.raises(ValueError, match="Unsupported SDK response schema"):
        ReplayJudge("retained").generate("fixture", schema=Unsupported)


def test_live_requires_explicit_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="requires OPENAI_API_KEY"):
        live_judge()


def test_actual_provider_adapter_constructs_without_requests(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder-not-a-secret")
    monkeypatch.setenv("CONVERSATION_JUDGE_MODEL", "gpt-4o-mini")
    from deepeval.models import OpenAIModel
    model = live_judge()
    assert isinstance(model, OpenAIModel)
    assert "gpt-4o-mini" in model.get_model_name()


def test_replay_cli_saves_json_and_failure_exit(tmp_path, capsys):
    output = tmp_path / "report.json"
    assert main(["--case", "forgotten", "--report", str(output)]) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert json.loads(capsys.readouterr().out)["mode"] == "scripted_replay"


def test_configuration_exit_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as result:
        main(["--live"])
    assert result.value.code == 2
