"""Chat-completions client and prompt helpers for the LLM comparison runs.

Extracted verbatim from ``graph_jepa_v{5,6}/evaluate_llm.py``, where these six
definitions were duplicated. The two copies are byte-identical over this region,
so the dedup needed no reconciliation; the v6 copy was taken.

This module imports no first-party code, and must not start to. It is the
boundary-safe half of ``benchmarks``: ``vs_llm.py`` and ``vs_fawkes.py``
import both model lineages, this does not, so an LLM run can be exercised
without loading torch or either package.

``_load_dotenv_files`` and ``_api_key`` locate the repository root with
``Path(__file__).resolve().parents[2]``. That depth is unchanged by the move:
``src/benchmarks/llm_ranker.py`` sits exactly as deep as
``old_src/graph_jepa_v6/evaluate_llm.py`` did.

Requires ``openai`` and ``python-dotenv``, both imported lazily at the point of
use so that importing this module costs nothing when no LLM run is happening.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def _load_dotenv_files() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")
    load_dotenv()


def _api_base(provider: str) -> str:
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1"
    if provider == "cerebras":
        return "https://api.cerebras.ai/v1"
    raise ValueError(f"unknown provider: {provider!r}")


def _api_key(provider: str) -> str:
    _load_dotenv_files()
    env_name = "OPENROUTER_API_KEY" if provider == "openrouter" else "CEREBRAS_API_KEY"
    key = os.environ.get(env_name, "").strip()
    if key:
        return key

    for path in (
        Path("api_keys.json"),
        Path(__file__).resolve().parents[2] / "api_keys.json",
    ):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        key = str(data.get(provider) or "").strip()
        if key:
            return key

    raise SystemExit(f"Set {env_name} in .env or add {provider!r} to api_keys.json.")


def _node_text(node: dict) -> str:
    return str(
        node.get("text")
        or node.get("normalized_name")
        or node.get("name")
        or node.get("id")
        or ""
    ).strip()


def _node_label(graph, node_idx: int) -> str:
    node = graph.nodes[node_idx]
    return f"{node.get('type', '')}: {_node_text(node)}"


class ChatRanker:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        reasoning_effort: str | None,
        reasoning_format: str | None,
        retries: int,
        sleep: float,
    ):
        from openai import OpenAI

        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.reasoning_format = reasoning_format
        self.retries = retries
        self.sleep = sleep
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)

    def rank(self, prompt: str) -> tuple[str, dict[str, int | str]]:
        last_error = None
        for attempt in range(self.retries):
            try:
                extra_body = {}
                if self.reasoning_format:
                    extra_body["reasoning_format"] = self.reasoning_format
                kwargs = {}
                if self.reasoning_effort:
                    kwargs["reasoning_effort"] = self.reasoning_effort
                if extra_body:
                    kwargs["extra_body"] = extra_body
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **kwargs,
                )
                usage = {}
                if completion.usage:
                    prompt_tokens = completion.usage.prompt_tokens or 0
                    completion_tokens = completion.usage.completion_tokens or 0
                    total_tokens = getattr(completion.usage, "total_tokens", None)
                    completion_details = getattr(
                        completion.usage,
                        "completion_tokens_details",
                        None,
                    )
                    reasoning_tokens = (
                        getattr(completion_details, "reasoning_tokens", 0)
                        if completion_details
                        else 0
                    )
                    usage = {
                        "prompt_tokens": int(prompt_tokens),
                        "completion_tokens": int(completion_tokens),
                        "reasoning_tokens": int(reasoning_tokens or 0),
                        "total_tokens": int(
                            total_tokens
                            if total_tokens is not None
                            else prompt_tokens + completion_tokens
                        ),
                        "finish_reason": str(completion.choices[0].finish_reason or ""),
                    }
                return completion.choices[0].message.content or "", usage
            except Exception as exc:
                last_error = exc
                print(
                    f"[LLM] retry {attempt + 1}/{self.retries} failed: {exc}",
                    flush=True,
                )
                time.sleep(self.sleep * (2**attempt))
        print(
            f"[LLM] giving up after {self.retries} attempts: {last_error}",
            flush=True,
        )
        return "", {}
