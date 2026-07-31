"""Load `.env` into os.environ, in-process. Used by scripts only.

Why this exists rather than `set -a; . ./.env; set +a`: that shell idiom exports
the whole file, so any command whose output is shared prints every value in it —
which is how a LangSmith key ended up in a transcript. Loading in-process keeps
the values out of the shell and out of any log.

Why it lives in `scripts/` and not in `src/config.py`: `config.py` reads
`os.environ` only, deliberately. A config module that silently absorbs a dotfile
makes it impossible to say what the scored run actually used, and the scored run's
environment has to be explicit. `langchain-basics` calls `load_dotenv()` at its
entry points for the same separation — the convenience belongs to the entry point,
not the library.

No dependency: python-dotenv is not in requirements.txt and this is ~20 lines.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path | None = None, *, override: bool = False) -> list[str]:
    """Load KEY=VALUE lines from `.env`. Returns the names loaded, never values.

    Real environment variables win by default, so `LANGSMITH_TRACING=true
    .venv/bin/python scripts/...` still beats whatever the file says.
    """
    path = path or REPO_ROOT / ".env"
    if not path.is_file():
        return []

    loaded: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if not key or (not override and key in os.environ):
            continue
        # An empty value in .env means "unset" -- notably LANGSMITH_WORKSPACE_ID,
        # where a stale value makes every LangSmith READ return 403 while writes
        # still succeed.
        if value:
            os.environ[key] = value
            loaded.append(key)
        else:
            os.environ.pop(key, None)
    return loaded
