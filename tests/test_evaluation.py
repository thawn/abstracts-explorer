"""
Tests for the evaluation module.

This module tests:
- Database models (EvalQAPair, EvalResult)
- Database CRUD methods for evaluation data
- Evaluator class (Q/A pair generation, evaluation runner)
- Result formatting
- CLI eval sub-commands
"""

import json
import sys
from unittest.mock import Mock, patch

import pytest

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.evaluation import (
    EvaluationError,
    Evaluator,
    _parse_json_array,
    _sample_papers_context,
    format_eval_result_detail,
    format_eval_summary,
)
from abstracts_explorer.cli import main
from abstracts_explorer.plugin import LightweightPaper
from tests.conftest import set_test_db

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def eval_db(tmp_path, monkeypatch):
    """Create a connected database with tables for evaluation tests."""
    db_path = tmp_path / "eval_test.db"
    set_test_db(db_path)

    db = DatabaseManager()
    db.connect()
    db.create_tables()
    yield db
    db.close()


@pytest.fixture
def eval_db_with_papers(eval_db):
    """Database with sample papers loaded."""
    papers = [
        LightweightPaper(
            title="Deep Learning for Image Classification",
            authors=["Alice Smith", "Bob Jones"],
            abstract="We present a novel deep learning approach for image classification.",
            session="Session A",
            poster_position="A1",
            year=2025,
            conference="NeurIPS",
            keywords=["deep learning", "image classification"],
        ),
        LightweightPaper(
            title="Reinforcement Learning in Robotics",
            authors=["Charlie Brown"],
            abstract="This paper explores reinforcement learning applications in robotics.",
            session="Session B",
            poster_position="B1",
            year=2025,
            conference="NeurIPS",
            keywords=["reinforcement learning", "robotics"],
        ),
        LightweightPaper(
            title="Transformer Models for NLP",
            authors=["Diana Prince"],
            abstract="We analyze transformer architectures for natural language processing.",
            session="Session C",
            poster_position="C1",
            year=2024,
            conference="ICLR",
            keywords=["transformers", "NLP"],
        ),
    ]
    eval_db.add_papers(papers)
    return eval_db


@pytest.fixture
def eval_db_with_pairs(eval_db):
    """Database with sample Q/A pairs."""
    pairs = [
        {
            "conversation_id": "conv001",
            "turn_number": 0,
            "query": "What are the main topics at NeurIPS 2025?",
            "expected_answer": "The main topics include deep learning and reinforcement learning.",
            "tool_name": "get_cluster_topics",
        },
        {
            "conversation_id": "conv001",
            "turn_number": 1,
            "query": "Tell me more about deep learning papers.",
            "expected_answer": "There is a paper on deep learning for image classification.",
            "tool_name": "search_papers",
        },
        {
            "conversation_id": "conv002",
            "turn_number": 0,
            "query": "How has transformer research evolved?",
            "expected_answer": "Transformer research has grown significantly.",
            "tool_name": "get_topic_evolution",
        },
    ]
    for p in pairs:
        eval_db.add_eval_qa_pair(**p)
    return eval_db


@pytest.fixture
def verified_eval_db(eval_db_with_pairs):
    """Database with verified Q/A pairs."""
    pairs = eval_db_with_pairs.get_eval_qa_pairs()
    for p in pairs:
        eval_db_with_pairs.update_eval_qa_pair(p["id"], verified=1)
    return eval_db_with_pairs


@pytest.fixture
def mock_em():
    """A mock EmbeddingsManager with a mock openai_client."""
    em = Mock()
    em.openai_client = Mock()
    return em


@pytest.fixture
def evaluator_with_papers(eval_db_with_papers, mock_em):
    """Evaluator backed by a database with papers and a mocked embeddings manager."""
    return Evaluator(embeddings_manager=mock_em, db=eval_db_with_papers)


@pytest.fixture
def evaluator_verified(verified_eval_db, mock_em):
    """Evaluator backed by a database with verified Q/A pairs."""
    return Evaluator(embeddings_manager=mock_em, db=verified_eval_db)


# ---------------------------------------------------------------------------
#  Tests: Database Models
# ---------------------------------------------------------------------------


class TestEvalQAPairModel:
    """Tests for EvalQAPair database model."""

    def test_add_eval_qa_pair(self, eval_db):
        """Test inserting a Q/A pair."""
        pair_id = eval_db.add_eval_qa_pair(
            conversation_id="test-conv",
            turn_number=0,
            query="What papers are about deep learning?",
            expected_answer="Several papers cover deep learning.",
            tool_name="search_papers",
            source_info='{"model": "test"}',
        )

        assert pair_id is not None
        assert isinstance(pair_id, int)

    def test_get_eval_qa_pairs(self, eval_db):
        """Test retrieving Q/A pairs."""
        eval_db.add_eval_qa_pair(
            conversation_id="c1",
            turn_number=0,
            query="Query 1",
            expected_answer="Answer 1",
            tool_name="search_papers",
        )
        eval_db.add_eval_qa_pair(
            conversation_id="c2",
            turn_number=0,
            query="Query 2",
            expected_answer="Answer 2",
            tool_name="get_cluster_topics",
        )

        pairs = eval_db.get_eval_qa_pairs()
        assert len(pairs) == 2
        assert pairs[0]["query"] == "Query 1"
        assert pairs[1]["query"] == "Query 2"

    def test_get_eval_qa_pairs_filter_tool(self, eval_db_with_pairs):
        """Test filtering Q/A pairs by tool name."""
        pairs = eval_db_with_pairs.get_eval_qa_pairs(tool_name="search_papers")
        assert len(pairs) == 1
        assert pairs[0]["tool_name"] == "search_papers"

    def test_get_eval_qa_pairs_filter_verified(self, eval_db_with_pairs):
        """Test filtering verified-only Q/A pairs."""
        # All are unverified initially
        pairs = eval_db_with_pairs.get_eval_qa_pairs(verified_only=True)
        assert len(pairs) == 0

        # Verify one pair
        all_pairs = eval_db_with_pairs.get_eval_qa_pairs()
        eval_db_with_pairs.update_eval_qa_pair(all_pairs[0]["id"], verified=1)

        pairs = eval_db_with_pairs.get_eval_qa_pairs(verified_only=True)
        assert len(pairs) == 1

    def test_get_eval_qa_pairs_filter_conversation(self, eval_db_with_pairs):
        """Test filtering Q/A pairs by conversation_id."""
        pairs = eval_db_with_pairs.get_eval_qa_pairs(conversation_id="conv001")
        assert len(pairs) == 2
        assert all(p["conversation_id"] == "conv001" for p in pairs)
        # Ordered by turn_number
        assert pairs[0]["turn_number"] == 0
        assert pairs[1]["turn_number"] == 1

    def test_get_eval_qa_pairs_limit_offset(self, eval_db_with_pairs):
        """Test pagination with limit and offset."""
        pairs = eval_db_with_pairs.get_eval_qa_pairs(limit=2)
        assert len(pairs) == 2

        pairs = eval_db_with_pairs.get_eval_qa_pairs(limit=2, offset=2)
        assert len(pairs) == 1

    def test_get_eval_qa_pair_count(self, eval_db_with_pairs):
        """Test counting Q/A pairs."""
        count = eval_db_with_pairs.get_eval_qa_pair_count()
        assert count == 3

    def test_get_eval_qa_pair_count_verified(self, eval_db_with_pairs):
        """Test counting verified-only Q/A pairs."""
        count = eval_db_with_pairs.get_eval_qa_pair_count(verified_only=True)
        assert count == 0

        pairs = eval_db_with_pairs.get_eval_qa_pairs()
        eval_db_with_pairs.update_eval_qa_pair(pairs[0]["id"], verified=1)

        count = eval_db_with_pairs.get_eval_qa_pair_count(verified_only=True)
        assert count == 1

    def test_update_eval_qa_pair(self, eval_db):
        """Test updating a Q/A pair."""
        pair_id = eval_db.add_eval_qa_pair(
            conversation_id="c1",
            turn_number=0,
            query="Old query",
            expected_answer="Old answer",
        )

        result = eval_db.update_eval_qa_pair(pair_id, query="New query", verified=1)
        assert result is True

        pairs = eval_db.get_eval_qa_pairs()
        assert pairs[0]["query"] == "New query"
        assert pairs[0]["verified"] == 1

    def test_update_eval_qa_pair_not_found(self, eval_db):
        """Test updating a non-existent pair returns False."""
        result = eval_db.update_eval_qa_pair(99999, query="New query")
        assert result is False

    def test_update_eval_qa_pair_no_valid_fields(self, eval_db):
        """Test updating with no valid fields returns False."""
        pair_id = eval_db.add_eval_qa_pair(
            conversation_id="c1",
            turn_number=0,
            query="Query",
            expected_answer="Answer",
        )
        result = eval_db.update_eval_qa_pair(pair_id, invalid_field="value")
        assert result is False

    def test_delete_eval_qa_pair(self, eval_db):
        """Test deleting a Q/A pair."""
        pair_id = eval_db.add_eval_qa_pair(
            conversation_id="c1",
            turn_number=0,
            query="To delete",
            expected_answer="Will be deleted",
        )

        assert eval_db.get_eval_qa_pair_count() == 1
        result = eval_db.delete_eval_qa_pair(pair_id)
        assert result is True
        assert eval_db.get_eval_qa_pair_count() == 0

    def test_delete_eval_qa_pair_not_found(self, eval_db):
        """Test deleting a non-existent pair returns False."""
        result = eval_db.delete_eval_qa_pair(99999)
        assert result is False


# ---------------------------------------------------------------------------
#  Tests: EvalResult database methods
# ---------------------------------------------------------------------------


class TestEvalResultModel:
    """Tests for EvalResult database operations."""

    def test_add_eval_result(self, eval_db):
        """Test inserting an evaluation result."""
        result_id = eval_db.add_eval_result(
            run_id="run-001",
            qa_pair_id=1,
            actual_answer="This is the actual answer.",
            actual_tool_name="search_papers",
            answer_score=4.0,
            tool_correct=1,
            latency_ms=500,
            judge_reasoning="Good answer.",
        )

        assert result_id is not None
        assert isinstance(result_id, int)

    def test_get_eval_results(self, eval_db):
        """Test retrieving evaluation results."""
        eval_db.add_eval_result(run_id="run-001", qa_pair_id=1, answer_score=4.0)
        eval_db.add_eval_result(run_id="run-001", qa_pair_id=2, answer_score=3.0)
        eval_db.add_eval_result(run_id="run-002", qa_pair_id=3, answer_score=5.0)

        # All results
        results = eval_db.get_eval_results()
        assert len(results) == 3

        # Filter by run
        results = eval_db.get_eval_results(run_id="run-001")
        assert len(results) == 2

    def test_get_eval_results_pagination(self, eval_db):
        """Test pagination of evaluation results."""
        for i in range(5):
            eval_db.add_eval_result(run_id="run-001", qa_pair_id=i, answer_score=float(i))

        results = eval_db.get_eval_results(limit=3)
        assert len(results) == 3

        results = eval_db.get_eval_results(limit=3, offset=3)
        assert len(results) == 2

    def test_get_eval_run_ids(self, eval_db):
        """Test getting distinct run IDs."""
        eval_db.add_eval_result(run_id="run-002", qa_pair_id=1)
        eval_db.add_eval_result(run_id="run-001", qa_pair_id=2)

        run_ids = eval_db.get_eval_run_ids()
        assert isinstance(run_ids, list)
        assert len(run_ids) == 2
        assert "run-001" in run_ids
        assert "run-002" in run_ids

    def test_get_eval_run_summary(self, eval_db):
        """Test computing run summary statistics."""
        eval_db.add_eval_result(
            run_id="run-001",
            qa_pair_id=1,
            answer_score=4.0,
            tool_correct=1,
            latency_ms=500,
        )
        eval_db.add_eval_result(
            run_id="run-001",
            qa_pair_id=2,
            answer_score=3.0,
            tool_correct=0,
            latency_ms=700,
        )
        eval_db.add_eval_result(
            run_id="run-001",
            qa_pair_id=3,
            error="timeout",
            latency_ms=5000,
        )

        summary = eval_db.get_eval_run_summary("run-001")
        assert summary["total"] == 3
        assert summary["avg_score"] == pytest.approx(3.5)
        assert summary["tool_accuracy"] == pytest.approx(0.5)
        assert summary["avg_latency_ms"] == pytest.approx(2066.67, abs=1)
        assert summary["error_count"] == 1

    def test_get_eval_run_summary_empty(self, eval_db):
        """Test summary for non-existent run."""
        summary = eval_db.get_eval_run_summary("nonexistent")
        assert summary["total"] == 0
        assert summary["avg_score"] is None
        assert summary["tool_accuracy"] is None
        assert summary["avg_latency_ms"] is None
        assert summary["error_count"] == 0


# ---------------------------------------------------------------------------
#  Tests: Evaluation module functions
# ---------------------------------------------------------------------------


class TestParseJsonArray:
    """Tests for _parse_json_array helper."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON array."""
        text = '[{"query": "test", "expected_answer": "answer"}]'
        result = _parse_json_array(text)
        assert len(result) == 1
        assert result[0]["query"] == "test"

    def test_parse_json_with_code_fences(self):
        """Test parsing JSON wrapped in markdown code fences."""
        text = '```json\n[{"query": "test", "expected_answer": "answer"}]\n```'
        result = _parse_json_array(text)
        assert len(result) == 1

    def test_parse_json_with_think_blocks(self):
        """Test parsing JSON with <think> blocks stripped."""
        text = '<think>Let me think...</think>[{"query": "test", "expected_answer": "answer"}]'
        result = _parse_json_array(text)
        assert len(result) == 1

    def test_parse_invalid_json_raises(self):
        """Test that invalid JSON raises EvaluationError."""
        with pytest.raises(EvaluationError, match="Failed to parse"):
            _parse_json_array("not valid json")

    def test_parse_non_array_raises(self):
        """Test that non-array JSON raises EvaluationError."""
        with pytest.raises(EvaluationError, match="Expected JSON array"):
            _parse_json_array('{"key": "value"}')


class TestSamplePapersContext:
    """Tests for _sample_papers_context helper."""

    def test_sample_papers_from_db(self, eval_db_with_papers):
        """Test sampling papers for context."""
        context = _sample_papers_context(eval_db_with_papers, n_papers=2)
        assert "Paper" in context
        assert len(context) > 0

    def test_sample_papers_empty_db(self, eval_db):
        """Test sampling from empty database."""
        context = _sample_papers_context(eval_db)
        assert "no papers" in context


class TestEvaluatorClass:
    """Tests for the Evaluator class constructor and properties."""

    def test_evaluator_requires_embeddings_manager(self, eval_db):
        """Test that Evaluator raises when embeddings_manager is None."""
        with pytest.raises(EvaluationError, match="embeddings_manager is required"):
            Evaluator(embeddings_manager=None, db=eval_db)

    def test_evaluator_requires_db(self, mock_em):
        """Test that Evaluator raises when db is None."""
        with pytest.raises(EvaluationError, match="db is required"):
            Evaluator(embeddings_manager=mock_em, db=None)

    def test_evaluator_openai_client_delegates(self, eval_db, mock_em):
        """Test that openai_client property delegates to EmbeddingsManager."""
        evaluator = Evaluator(embeddings_manager=mock_em, db=eval_db)
        assert evaluator.openai_client is mock_em.openai_client


class TestGenerateQAPairs:
    """Tests for Evaluator.generate_qa_pairs method."""

    def test_generate_qa_pairs_basic(self, evaluator_with_papers):
        """Test basic Q/A pair generation with mocked LLM."""
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        [
                            {
                                "query": "What papers discuss deep learning?",
                                "expected_answer": "There is a paper about deep learning for image classification.",
                            }
                        ]
                    )
                )
            )
        ]

        evaluator_with_papers.openai_client.chat.completions.create.return_value = mock_response

        pairs = evaluator_with_papers.generate_qa_pairs(
            n_pairs_per_tool=1,
            tools=["search_papers"],
            generate_followups=False,
        )

        assert len(pairs) == 1
        assert pairs[0]["tool_name"] == "search_papers"
        assert pairs[0]["turn_number"] == 0
        assert pairs[0]["query"] == "What papers discuss deep learning?"

    def test_generate_qa_pairs_with_followups(self, evaluator_with_papers):
        """Test Q/A pair generation with follow-ups."""
        # Initial pair response
        initial_response = Mock()
        initial_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        [{"query": "What are main topics?", "expected_answer": "Deep learning and RL."}]
                    )
                )
            )
        ]

        # Follow-up response
        followup_response = Mock()
        followup_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        [
                            {
                                "query": "Tell me more about RL papers.",
                                "expected_answer": "There is a paper on RL in robotics.",
                            }
                        ]
                    )
                )
            )
        ]

        evaluator_with_papers.openai_client.chat.completions.create.side_effect = [
            initial_response,
            followup_response,
        ]

        pairs = evaluator_with_papers.generate_qa_pairs(
            n_pairs_per_tool=1,
            tools=["get_cluster_topics"],
            generate_followups=True,
            n_followups=1,
        )

        assert len(pairs) == 2
        assert pairs[0]["turn_number"] == 0
        assert pairs[1]["turn_number"] == 1
        assert pairs[0]["conversation_id"] == pairs[1]["conversation_id"]

    def test_generate_qa_pairs_unknown_tool(self, evaluator_with_papers):
        """Test that unknown tool names raise an error."""
        with pytest.raises(EvaluationError, match="Unknown tool"):
            evaluator_with_papers.generate_qa_pairs(
                tools=["nonexistent_tool"],
            )

    def test_generate_qa_pairs_llm_failure(self, evaluator_with_papers):
        """Test graceful handling of LLM failure."""
        evaluator_with_papers.openai_client.chat.completions.create.side_effect = Exception("API timeout")

        with pytest.raises(EvaluationError, match="LLM call failed"):
            evaluator_with_papers.generate_qa_pairs(
                tools=["search_papers"],
                generate_followups=False,
            )


class TestStoreQAPairs:
    """Tests for Evaluator.store_qa_pairs method."""

    def test_store_pairs(self, eval_db, mock_em):
        """Test persisting Q/A pairs to database."""
        pairs = [
            {
                "conversation_id": "c1",
                "turn_number": 0,
                "query": "Query 1",
                "expected_answer": "Answer 1",
                "tool_name": "search_papers",
                "source_info": "{}",
            },
            {
                "conversation_id": "c1",
                "turn_number": 1,
                "query": "Follow-up",
                "expected_answer": "Follow-up answer",
                "tool_name": "search_papers",
                "source_info": "{}",
            },
        ]

        evaluator = Evaluator(embeddings_manager=mock_em, db=eval_db)
        count = evaluator.store_qa_pairs(pairs)
        assert count == 2
        assert eval_db.get_eval_qa_pair_count() == 2


# ---------------------------------------------------------------------------
#  Tests: Evaluation runner
# ---------------------------------------------------------------------------


class TestRunEvaluation:
    """Tests for the Evaluator.run_evaluation method."""

    def test_run_evaluation_no_pairs(self, eval_db, mock_em):
        """Test that running with no pairs raises an error."""
        evaluator = Evaluator(embeddings_manager=mock_em, db=eval_db)

        with pytest.raises(EvaluationError, match="No Q/A pairs found"):
            evaluator.run_evaluation()

    def test_run_evaluation_basic(self, evaluator_verified):
        """Test basic evaluation run with mocked RAG and judge."""
        # Mock RAG query result
        mock_rag_result = {
            "response": "Deep learning and RL are the main topics.",
            "papers": [],
            "metadata": {
                "tools_executed": ["get_cluster_topics"],
                "n_papers": 0,
            },
        }

        # Mock judge response
        mock_judge_response = Mock()
        mock_judge_response.choices = [Mock(message=Mock(content='{"score": 4, "reasoning": "Good answer."}'))]
        evaluator_verified.openai_client.chat.completions.create.return_value = mock_judge_response

        with patch("abstracts_explorer.rag.RAGChat") as MockRAG:
            mock_rag_instance = MockRAG.return_value
            mock_rag_instance.query.return_value = mock_rag_result

            run_id = evaluator_verified.run_evaluation(verified_only=True)

        assert run_id.startswith("eval-")

        # Verify results were stored
        results = evaluator_verified.db.get_eval_results(run_id=run_id)
        assert len(results) == 3  # 3 verified pairs

        # Check scores
        for r in results:
            assert r["answer_score"] == 4.0
            assert r["actual_answer"] is not None

    def test_run_evaluation_with_error(self, evaluator_verified):
        """Test evaluation handles query errors gracefully."""
        with patch("abstracts_explorer.rag.RAGChat") as MockRAG:
            mock_rag_instance = MockRAG.return_value
            mock_rag_instance.query.side_effect = Exception("Connection timeout")

            run_id = evaluator_verified.run_evaluation()

        results = evaluator_verified.db.get_eval_results(run_id=run_id)
        assert len(results) == 3
        for r in results:
            assert r["error"] is not None
            assert "timeout" in r["error"]


# ---------------------------------------------------------------------------
#  Tests: Result formatting
# ---------------------------------------------------------------------------


class TestFormatting:
    """Tests for result formatting functions."""

    def test_format_eval_summary(self):
        """Test summary formatting."""
        summary = {
            "total": 10,
            "avg_score": 3.75,
            "tool_accuracy": 0.8,
            "avg_latency_ms": 1200.0,
            "error_count": 1,
        }
        output = format_eval_summary(summary, "eval-abc123")
        assert "eval-abc123" in output
        assert "10" in output
        assert "3.75" in output
        assert "80.0%" in output
        assert "1200" in output
        assert "1" in output

    def test_format_eval_summary_no_data(self):
        """Test summary formatting with no data."""
        summary = {
            "total": 0,
            "avg_score": None,
            "tool_accuracy": None,
            "avg_latency_ms": None,
            "error_count": 0,
        }
        output = format_eval_summary(summary, "eval-empty")
        assert "N/A" in output
        assert "0" in output

    def test_format_eval_result_detail(self):
        """Test detail formatting with Q/A pair."""
        result = {
            "id": 1,
            "run_id": "eval-abc",
            "qa_pair_id": 5,
            "actual_answer": "The main topics are deep learning and transformers.",
            "actual_tool_name": "get_cluster_topics",
            "answer_score": 4.0,
            "tool_correct": 1,
            "latency_ms": 800,
            "error": None,
            "judge_reasoning": "Good coverage of topics.",
        }
        qa_pair = {
            "id": 5,
            "query": "What are the main topics?",
            "expected_answer": "Deep learning and transformers are the main topics.",
            "tool_name": "get_cluster_topics",
        }
        output = format_eval_result_detail(result, qa_pair)
        assert "What are the main topics?" in output
        assert "4.0/5" in output
        assert "✅ correct" in output
        assert "800 ms" in output

    def test_format_eval_result_detail_without_qa_pair(self):
        """Test detail formatting without Q/A pair."""
        result = {
            "id": 1,
            "run_id": "eval-abc",
            "qa_pair_id": 5,
            "actual_answer": "Some answer",
            "actual_tool_name": None,
            "answer_score": None,
            "tool_correct": None,
            "latency_ms": None,
            "error": "Connection failed",
            "judge_reasoning": None,
        }
        output = format_eval_result_detail(result, None)
        assert "QA Pair ID: 5" in output
        assert "Connection failed" in output
        assert "N/A" in output


# ---------------------------------------------------------------------------
#  Tests: CLI commands
# ---------------------------------------------------------------------------


class TestCLIEvalCommands:
    """Tests for eval CLI sub-commands."""

    def test_eval_no_subcommand(self, capsys):
        """Test 'eval' with no sub-command shows help."""
        with patch.object(sys, "argv", ["abstracts-explorer", "eval"]):
            exit_code = main()
            assert exit_code == 1
            captured = capsys.readouterr()
            assert "generate" in captured.out or "eval" in captured.out

    def test_eval_generate_no_papers(self, tmp_path, capsys):
        """Test eval generate with empty database."""
        db_path = tmp_path / "empty.db"
        set_test_db(db_path)

        with patch.object(
            sys,
            "argv",
            ["abstracts-explorer", "eval", "generate", "--tools", "search_papers", "--no-followups"],
        ):
            exit_code = main()
            assert exit_code == 1
            captured = capsys.readouterr()
            assert "No papers" in captured.err

    def test_eval_generate_success(self, eval_db_with_papers, capsys):
        """Test successful Q/A pair generation via CLI."""
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content=json.dumps([{"query": "Test query", "expected_answer": "Test answer"}])))
        ]
        mock_openai_client = Mock()
        mock_openai_client.chat.completions.create.return_value = mock_response

        mock_em_instance = Mock()
        mock_em_instance.openai_client = mock_openai_client

        with (
            patch("abstracts_explorer.cli.EmbeddingsManager", return_value=mock_em_instance),
            patch.object(
                sys,
                "argv",
                [
                    "abstracts-explorer",
                    "eval",
                    "generate",
                    "--tools",
                    "search_papers",
                    "--n-pairs",
                    "1",
                    "--no-followups",
                ],
            ),
        ):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Generated and stored" in captured.out

    def test_eval_results_no_runs(self, eval_db, capsys):
        """Test eval results with no evaluation runs."""
        with patch.object(sys, "argv", ["abstracts-explorer", "eval", "results"]):
            exit_code = main()
            assert exit_code == 0
            captured = capsys.readouterr()
            assert "No evaluation results" in captured.out

    def test_eval_results_with_data(self, eval_db_with_pairs, capsys):
        """Test eval results display."""
        # Add some results
        eval_db_with_pairs.add_eval_result(
            run_id="eval-test",
            qa_pair_id=1,
            answer_score=4.0,
            tool_correct=1,
            latency_ms=500,
        )

        with patch.object(
            sys,
            "argv",
            ["abstracts-explorer", "eval", "results", "--run-id", "eval-test"],
        ):
            exit_code = main()
            assert exit_code == 0
            captured = capsys.readouterr()
            assert "eval-test" in captured.out

    def test_eval_results_with_detail(self, eval_db_with_pairs, capsys):
        """Test eval results with detail flag."""
        eval_db_with_pairs.add_eval_result(
            run_id="eval-test",
            qa_pair_id=1,
            actual_answer="Test answer",
            answer_score=4.0,
        )

        with patch.object(
            sys,
            "argv",
            ["abstracts-explorer", "eval", "results", "--detail"],
        ):
            exit_code = main()
            assert exit_code == 0
            captured = capsys.readouterr()
            assert "Result #" in captured.out
