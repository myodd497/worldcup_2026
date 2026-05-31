"""
Docs Agent — logs every user session to bin/docs/ as a markdown file.
"""
from __future__ import annotations

import datetime
from pathlib import Path

_DOCS_DIR = Path(__file__).resolve().parents[2] / "bin" / "docs"


def log_session(user_id: str, user_message: str, agent_response: str) -> None:
    _DOCS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_uid = user_id.replace(":", "_").replace("+", "")
    file_path = _DOCS_DIR / f"session_{safe_uid}_{ts}.md"

    content = f"""# Session Log — {ts}

**User:** `{user_id}`
**Message:** {user_message}

## Response
{agent_response}
"""
    file_path.write_text(content, encoding="utf-8")
