"""Loopback-only LLM report labeler (PhysioNet-compliant).

MIMIC-CXR reports are PhysioNet credentialed data and must not be sent to external LLM APIs. This
wrapper only talks to an inference server on the local machine or a private network address
(vLLM / Ollama / llama.cpp), refuses anything else, uses deterministic decoding, and parses a strict
JSON answer into CheXpert-style labels. Network calls happen only in :meth:`LocalLLMLabeler.label`.
"""
from __future__ import annotations

import ipaddress
import json
import re
import socket
from dataclasses import dataclass, field
from urllib.parse import urlparse

from .report_labeler import FINDINGS

LABEL_VALUES = {"positive": 1, "negative": 0, "uncertain": -1, "not_mentioned": None}


def assert_local_endpoint(base_url: str, allow_private_networks: bool = True) -> None:
    """Raise ``ValueError`` unless ``base_url`` resolves to loopback (or an RFC1918/ULA address)."""
    u = urlparse(base_url)
    host = u.hostname
    if not host:
        raise ValueError("invalid endpoint URL")
    if host in ("localhost",):
        return
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(host))
        except (socket.gaierror, ValueError) as e:
            raise ValueError(f"cannot resolve {host!r}; refusing non-local endpoint") from e
    if ip.is_loopback:
        return
    if allow_private_networks and ip.is_private:
        return
    raise ValueError(f"endpoint {base_url} is not local/private; PhysioNet data may not be sent to external services")


def build_prompt(report: str, findings: list[str] | None = None) -> str:
    findings = findings or [f for f in FINDINGS if f != "No Finding"]
    schema = ", ".join(f'"{f}": "<positive|negative|uncertain|not_mentioned>"' for f in findings)
    return (
        "You are labeling a chest radiograph report. For each finding, answer strictly from the report text: "
        "'positive' if the report states it is present, 'negative' if explicitly absent, 'uncertain' if hedged, "
        "'not_mentioned' if the report does not address it. Respond with a single JSON object and nothing else.\n"
        f"JSON schema: {{{schema}}}\n\nREPORT:\n{report.strip()}\n\nJSON:"
    )


def parse_response(text: str, findings: list[str] | None = None) -> dict[str, int | None]:
    """Extract the first JSON object from ``text`` and map values to {1, 0, -1, None}."""
    findings = findings or [f for f in FINDINGS if f != "No Finding"]
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in model response")
    obj = json.loads(m.group(0))
    out: dict[str, int | None] = {}
    for f in findings:
        v = str(obj.get(f, "not_mentioned")).strip().lower().replace(" ", "_")
        if v not in LABEL_VALUES:
            v = "uncertain" if v.startswith(("poss", "maybe", "unc")) else "not_mentioned"
        out[f] = LABEL_VALUES[v]
    any_abn = any(v in (1, -1) for k, v in out.items() if k != "Support Devices")
    out["No Finding"] = None if any_abn else 1
    return out


@dataclass
class LocalLLMLabeler:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama3:70b"
    backend: str = "ollama"          # 'ollama' (/api/generate) or 'openai' (vLLM's /v1/chat/completions)
    temperature: float = 0.0
    timeout: int = 120
    findings: list[str] = field(default_factory=lambda: [f for f in FINDINGS if f != "No Finding"])

    def __post_init__(self) -> None:
        assert_local_endpoint(self.base_url)

    def _payload(self, prompt: str) -> tuple[str, dict]:
        if self.backend == "ollama":
            return (f"{self.base_url}/api/generate",
                    {"model": self.model, "prompt": prompt, "stream": False,
                     "options": {"temperature": self.temperature, "seed": 0}, "format": "json"})
        if self.backend == "openai":
            return (f"{self.base_url}/v1/chat/completions",
                    {"model": self.model, "temperature": self.temperature, "seed": 0,
                     "messages": [{"role": "user", "content": prompt}]})
        raise ValueError(self.backend)

    @staticmethod
    def _extract_text(backend: str, data: dict) -> str:
        if backend == "ollama":
            return data.get("response", "")
        return data["choices"][0]["message"]["content"]

    def label(self, report: str) -> dict[str, int | None]:
        """Send one report to the local server and return CheXpert-style labels."""
        import requests  # imported lazily so the module has no network dependency at import time

        assert_local_endpoint(self.base_url)
        url, payload = self._payload(build_prompt(report, self.findings))
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return parse_response(self._extract_text(self.backend, r.json()), self.findings)
