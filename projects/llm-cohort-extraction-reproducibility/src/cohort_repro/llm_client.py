"""Provider-agnostic chat-completion client with a local-inference default.

Policy (PhysioNet, "Use of MIMIC Data with Large Language Models and Online Services"): credentialed
data must not be shared with third-party services that retain or train on inputs; locally deployed
models are the recommended route. This module therefore

1. defaults to an OpenAI-compatible *local* server (vLLM, llama.cpp ``llama-server``, Ollama,
   LM Studio) at ``http://localhost:8000/v1``;
2. offers an in-process ``TransformersClient`` (Hugging Face) as a second local route;
3. refuses any non-local base URL unless *both* ``COHORT_REPRO_ALLOW_REMOTE=1`` is set in the
   environment and ``allow_remote=True`` is passed, so a remote endpoint can never be reached by
   accident;
4. never constructs prompts from MIMIC records: callers pass public paper text only.

Configuration via environment: ``COHORT_REPRO_BASE_URL``, ``COHORT_REPRO_MODEL``, ``COHORT_REPRO_API_KEY``
(local servers usually ignore the key; a placeholder is sent).
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Union
from urllib.parse import urlparse

import requests

__all__ = [
    "LLMClient",
    "OpenAICompatibleClient",
    "TransformersClient",
    "DryRunClient",
    "RemoteEndpointError",
    "is_local_url",
    "extract_json",
    "chat_json",
    "default_client",
    "DEFAULT_BASE_URL",
]

DEFAULT_BASE_URL = "http://localhost:8000/v1"
_LOCAL_HOSTNAMES = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}


class RemoteEndpointError(RuntimeError):
    """Raised when a non-local endpoint is configured without an explicit double opt-in."""


def is_local_url(url: str, allow_private_networks: bool = True) -> bool:
    """True for loopback hosts (and, optionally, RFC1918 private-network addresses)."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host in _LOCAL_HOSTNAMES or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.is_loopback:
        return True
    return bool(allow_private_networks and ip.is_private)


class LLMClient(Protocol):
    """Minimal interface used by the extraction pipeline."""

    model: str

    def complete(self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 2048, seed: Optional[int] = None, json_mode: bool = False) -> str: ...


class OpenAICompatibleClient:
    """Chat-completions client for any OpenAI-compatible server (local by default)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 600.0,
        max_retries: int = 3,
        allow_remote: bool = False,
        session: Optional[Any] = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("COHORT_REPRO_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("COHORT_REPRO_MODEL") or "local-model"
        self.api_key = api_key or os.environ.get("COHORT_REPRO_API_KEY") or "local"
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()
        if not is_local_url(self.base_url):
            env_ok = os.environ.get("COHORT_REPRO_ALLOW_REMOTE") == "1"
            if not (allow_remote and env_ok):
                raise RemoteEndpointError(
                    f"{self.base_url} is not a local endpoint. This project runs open-weight models locally "
                    "(PhysioNet LLM policy). To override for non-MIMIC content only, set COHORT_REPRO_ALLOW_REMOTE=1 "
                    "and pass allow_remote=True."
                )

    def complete(self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 2048, seed: Optional[int] = None, json_mode: bool = False) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self.session.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout)
                if getattr(r, "status_code", 200) >= 500:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                if getattr(r, "status_code", 200) == 400 and json_mode:
                    payload.pop("response_format", None)  # server without JSON mode: retry plain
                    continue
                body = r.json()
                return body["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(1.5**attempt)
        raise RuntimeError(f"local LLM request failed after {self.max_retries + 1} attempts: {last}")


class TransformersClient:
    """In-process generation with Hugging Face ``transformers`` (lazy import; local weights)."""

    def __init__(self, model_name_or_path: str, device_map: str = "auto", torch_dtype: str = "auto", trust_remote_code: bool = False) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("pip install torch transformers accelerate to use TransformersClient") from exc
        self._torch = torch
        self.model = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        self._model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)

    def complete(self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 2048, seed: Optional[int] = None, json_mode: bool = False) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        inputs = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self._model.device)
        if seed is not None:
            self._torch.manual_seed(seed)
        gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_tokens, "pad_token_id": self.tokenizer.eos_token_id}
        if temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": temperature})
        else:
            gen_kwargs["do_sample"] = False
        with self._torch.no_grad():
            out = self._model.generate(inputs, **gen_kwargs)
        return self.tokenizer.decode(out[0][inputs.shape[-1] :], skip_special_tokens=True)


@dataclass
class DryRunClient:
    """Returns canned responses (a list, cycled, or a callable of the user prompt); records calls."""

    responses: Union[Sequence[str], Callable[[str], str]]
    model: str = "dry-run"
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def complete(self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 2048, seed: Optional[int] = None, json_mode: bool = False) -> str:
        self.calls.append({"system": system, "user": user, "temperature": temperature, "seed": seed, "json_mode": json_mode})
        if callable(self.responses):
            return self.responses(user)
        return self.responses[(len(self.calls) - 1) % len(self.responses)]


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Dict[str, Any]:
    """Parse the first JSON object in a model response (fenced or bare)."""
    m = _FENCE_RE.search(text)
    candidates = [m.group(1)] if m else []
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON object found in model response")


def chat_json(
    client: LLMClient,
    system: str,
    user: str,
    validator: Optional[Callable[[Dict[str, Any]], List[str]]] = None,
    max_attempts: int = 3,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """Ask for JSON, validate, and feed validation errors back (bounded). Never feeds data results back."""
    prompt = user
    last_err = ""
    for _ in range(max_attempts):
        text = client.complete(system, prompt, temperature=temperature, max_tokens=max_tokens, seed=seed, json_mode=True)
        try:
            obj = extract_json(text)
        except ValueError as exc:
            last_err = str(exc)
            prompt = user + f"\n\nYour previous reply was not valid JSON ({last_err}). Reply with one JSON object only."
            continue
        problems = validator(obj) if validator else []
        if not problems:
            return obj
        last_err = "; ".join(problems)
        prompt = user + f"\n\nYour previous JSON had these problems: {last_err}. Fix them and reply with one JSON object only."
    raise ValueError(f"no valid JSON after {max_attempts} attempts: {last_err}")


def default_client() -> LLMClient:
    """The local OpenAI-compatible client configured from the environment (never a remote API)."""
    return OpenAICompatibleClient()
