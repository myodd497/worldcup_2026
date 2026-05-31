"""
Code Review Agent — reviews dynamically generated Python code.
Runs Ruff (linting) + mypy (type checks) then an LLM review pass.
Only used when agents generate executable Python snippets.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from langchain_openai import ChatOpenAI

_llm = ChatOpenAI(model="gpt-4o", temperature=0)


def review_code(code: str) -> dict[str, str]:
    """
    Returns a dict with keys: 'ruff', 'mypy', 'llm_review', 'approved' (bool).
    """
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        tmp = Path(f.name)

    ruff_result = subprocess.run(
        ["ruff", "check", str(tmp)], capture_output=True, text=True
    )
    mypy_result = subprocess.run(
        ["mypy", str(tmp), "--ignore-missing-imports"], capture_output=True, text=True
    )
    tmp.unlink(missing_ok=True)

    ruff_output = ruff_result.stdout.strip() or "No issues found."
    mypy_output = mypy_result.stdout.strip() or "No issues found."

    llm_prompt = (
        f"Review this Python code for security vulnerabilities and logical errors.\n"
        f"Ruff output: {ruff_output}\nmypy output: {mypy_output}\n\n"
        f"Code:\n```python\n{code}\n```\n"
        f"Reply with: APPROVED or REJECTED, followed by a one-sentence reason."
    )
    llm_review = _llm.invoke(llm_prompt).content.strip()
    approved = llm_review.upper().startswith("APPROVED")

    return {
        "ruff": ruff_output,
        "mypy": mypy_output,
        "llm_review": llm_review,
        "approved": approved,
    }


if __name__ == "__main__":
    sample = "import os\nprint(os.environ.get('SECRET'))"
    print(review_code(sample))
