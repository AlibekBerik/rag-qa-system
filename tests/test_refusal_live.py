"""Live refusal test: out-of-scope questions must be refused, not answered.

Runs only when a Groq key and a built index are available:

    GROQ_API_KEY=... pytest -m live
"""
import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY") or not Path("chroma_db").exists(),
        reason="needs GROQ_API_KEY and a built chroma_db (run build_index.py)",
    ),
]

OUT_OF_SCOPE = [
    "What is the capital of France?",
    "Who won the 2018 FIFA World Cup?",
    "How do I install Python on Windows?",
    "What is the boiling point of water in Kelvin?",
    "Who wrote Hamlet?",
]


@pytest.mark.parametrize("question", OUT_OF_SCOPE)
def test_out_of_scope_question_is_refused(question):
    from fastapi.testclient import TestClient

    import api

    answer = TestClient(api.app).post("/ask", params={"question": question}).json()["answer"]
    assert "enough information" in answer.lower(), answer
