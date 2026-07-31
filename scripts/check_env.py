"""Environment doctor: report every variable the agent needs and whether it works.

    .venv/bin/python scripts/check_env.py

Read-only and safe to run any time. It never prints a secret — only its length
and prefix — so the output is safe to paste into a chat or an issue.

Exit code 0 when everything the SCORED path needs is in place; 1 otherwise.
LangSmith is dev-only and never affects the exit code.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import config  # noqa: E402

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "


def mask(value: str) -> str:
    if not value:
        return "<empty>"
    return f"<{value[:8]}... {len(value)} chars>"


def get(url: str, timeout: float = 8.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read(400).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, str(e)
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def post_chat(base: str, model: str, timeout: float = 45.0) -> tuple[bool, str]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 8,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {config.LITELLM_KEY}"} if config.LITELLM_KEY else {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        return True, (d["choices"][0]["message"]["content"] or "").strip()[:40]
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:90]}"


def main() -> int:
    fatal = 0
    print("=" * 72)
    print("ENV DOCTOR")
    print("=" * 72)

    # ---------------- how config is being sourced ----------------
    print("\n[1] Where values come from")
    env_file = REPO_ROOT / ".env"
    print(f"{OK if env_file.is_file() else WARN} .env present: {env_file.is_file()}")
    print(
        f"{WARN} src/config.py reads os.environ ONLY — it does not parse .env.\n"
        f"        Export first:  set -a; . ./.env; set +a\n"
        f"        (but NOT in a command whose output you share — that prints your keys)"
    )

    # ---------------- brain ----------------
    print("\n[2] Qwen brain — REQUIRED for the scored path")
    print(f"{OK} LITELLM_BASE_URL = {config.LITELLM_BASE_URL}")
    print(f"{OK} BRAIN_MODEL      = {config.BRAIN_MODEL}")
    print(f"{OK if config.LITELLM_KEY else WARN} LITELLM_KEY      = {mask(config.LITELLM_KEY)}"
          f"{'' if config.LITELLM_KEY else '  (empty is fine on this cluster)'}")

    code, body = get(f"{config.LITELLM_BASE_URL}/models")
    if code == 200:
        try:
            aliases = [m["id"] for m in json.loads(body).get("data", [])]
        except Exception:  # noqa: BLE001
            aliases = []
        print(f"{OK} LiteLLM reachable, aliases: {aliases or 'unparsed'}")
        if config.BRAIN_MODEL not in aliases and aliases:
            print(f"{BAD} BRAIN_MODEL '{config.BRAIN_MODEL}' is not a registered alias")
            fatal += 1
    else:
        print(f"{BAD} LiteLLM /models -> {code} {body[:80]}")
        fatal += 1

    ok, reply = post_chat(config.LITELLM_BASE_URL, config.BRAIN_MODEL)
    print(f"{OK if ok else BAD} brain answers: {reply}")
    fatal += 0 if ok else 1

    # ---------------- domain model ----------------
    print("\n[3] Nemotron — REQUIRED before official evaluation")
    print(f"{OK} DOMAIN_FT_MODEL     = {config.DOMAIN_FT_MODEL}")
    print(f"{OK} DOMAIN_BASE_URL     = {config.DOMAIN_BASE_URL}"
          f"{'  (inherits LITELLM_BASE_URL)' if config.DOMAIN_BASE_URL == config.LITELLM_BASE_URL else ''}")
    mode = config.DOMAIN_PREDICT_MODE
    if mode == "llm":
        print(f"{OK} DOMAIN_PREDICT_MODE = llm")
    else:
        print(
            f"{WARN} DOMAIN_PREDICT_MODE = {mode}\n"
            f"        MUST be 'llm' before official evaluation. Shipping 'mock' forfeits\n"
            f"        the entire 30% fine-tuned-model-quality category."
        )
    ok, reply = post_chat(config.DOMAIN_BASE_URL, config.DOMAIN_FT_MODEL)
    print(f"{OK if ok else WARN} {config.DOMAIN_FT_MODEL} answers: {reply}")

    # ---------------- data ----------------
    print("\n[4] Corpus — REQUIRED")
    src = "DATASET_DIR env" if os.getenv("DATASET_DIR") else "auto-resolved (leave unset)"
    print(f"{OK} DATASET_DIR = {config.DATASET_DIR}   [{src}]")
    for label, path, kind in (
        ("RBA", config.RBA_PATH, "file"),
        ("ASX", config.ASX_DIR, "dir"),
        ("AFR", config.AFR_DIR, "dir"),
    ):
        good = path.is_file() if kind == "file" else path.is_dir()
        extra = ""
        if good and kind == "dir":
            extra = f"  ({len(list(path.glob('*.jsonl')))} files)"
        print(f"{OK if good else BAD} {label}: {path.name}{extra}")
        fatal += 0 if good else 1

    # ---------------- eval ----------------
    print("\n[5] Evaluation")
    print(f"{OK} EVAL_JUDGE_MODEL = {config.EVAL_JUDGE_MODEL}"
          f"{'' if config.EVAL_JUDGE_MODEL == config.BRAIN_MODEL else '  (should normally be the brain)'}")

    # ---------------- langsmith ----------------
    print("\n[6] LangSmith — DEV ONLY, never affects the scored path or exit code")
    if config.LANGSMITH_TRACING:
        print(
            f"{WARN} LANGSMITH_TRACING = true\n"
            f"        Fine for debugging public questions. Turn OFF for the scored run:\n"
            f"        Challenge_Brief.md § Rules allows only approved local services."
        )
    else:
        print(f"{OK} LANGSMITH_TRACING = false  (correct for scoring)")

    key = os.getenv("LANGSMITH_API_KEY", "")
    print(f"{OK if key else WARN} LANGSMITH_API_KEY = {mask(key)}")
    ws = os.getenv("LANGSMITH_WORKSPACE_ID", "")
    if ws:
        print(
            f"{BAD} LANGSMITH_WORKSPACE_ID is SET ({ws[:8]}...)\n"
            f"        Unset it. The SDK sends it as the tenant header, and a value that\n"
            f"        does not match your key's workspace makes every READ 403 while\n"
            f"        writes still succeed — a very confusing failure."
        )
    else:
        print(f"{OK} LANGSMITH_WORKSPACE_ID unset  (correct)")

    if key:
        code, _ = get("https://api.smith.langchain.com/api/v1/sessions?limit=1")
        print(f"{OK if code == 200 else WARN} LangSmith read check (unauthenticated probe) -> {code}")

    # ---------------- unused ----------------
    strays = [
        v for v in (
            "MODEL_NAME", "LANGGRAPH_ASSISTANT_ID", "LANGSMITH_DEPLOYMENT_NAME",
            "EMBEDDING_MODEL_NAME", "EMBEDDING_CACHE_DIR",
        ) if os.getenv(v)
    ]
    if strays:
        print(f"\n[7] Set but unused by this project: {', '.join(strays)}")
        if os.getenv("MODEL_NAME", "").startswith("gpt"):
            print(f"{WARN} MODEL_NAME names an external OpenAI model. Nothing here reads it,")
            print("        and using one during scoring would breach the local-services rule.")

    print("\n" + "=" * 72)
    print(f"{'READY' if fatal == 0 else f'{fatal} BLOCKING PROBLEM(S)'} for the scored path")
    print("=" * 72)
    return 0 if fatal == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
