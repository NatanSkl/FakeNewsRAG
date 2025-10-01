"""
Local LLM client for llama.cpp's OpenAI-compatible server.

Usage example:
    from common.llm_client import Llama, Mistral
    llm = Llama()  # Uses default Llama configuration
    mistral = Mistral()  # Uses default Mistral configuration
    resp = llm.chat([
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Say hi in 3 words."}
    ])
    print(resp.text)
"""
from __future__ import annotations
import os
import time
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import requests


@dataclass
class ChatResponse:
    text: str
    raw: Dict[str, Any]


class LocalLLM:
    """Base class for local LLM clients using llama.cpp server."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://127.0.0.1:8010/v1")
        # llama.cpp ignores model name but we keep it for compatibility
        self.model = model or os.getenv("LLM_MODEL", "local-llama")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "unused")
        self.timeout = timeout
        self._session = requests.Session()

    # --------------- public API ---------------
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ChatResponse:
        """Send chat completion request and return aggregated text.

        Args:
            messages: list of {role, content}
            temperature, max_tokens, top_p, stop: standard decoding params
            extra: forwarded to the server body (e.g., logprobs, seed)
        """
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False,
        }
        if stop:
            body["stop"] = stop
        if extra:
            body.update(extra)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        t0 = time.time()
        resp = self._session.post(url, headers=headers, data=json.dumps(body), timeout=self.timeout)
        dt = (time.time() - t0) * 1000
        try:
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"LLM request failed ({dt:.1f} ms): {e}\nBody: {getattr(resp, 'text', '')[:600]}")

        data = resp.json()
        text = "".join(choice.get("message", {}).get("content", "") for choice in data.get("choices", []))
        return ChatResponse(text=text, raw=data)

    # --------------- convenience helpers ---------------
    def simple(self, prompt: str, system: Optional[str] = None, **kw: Any) -> str:
        msgs: List[Dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.chat(msgs, **kw).text


class Llama(LocalLLM):
    """Llama model client with default configuration.
    
    Assumes Llama server is running with:
        python -m llama_cpp.server \
          --model ./models/llama32-3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
          --n_gpu_layers -1 --n_ctx 2048 --n_batch 256 --offload_kqv true \
          --host 127.0.0.1 --port 8010
    """
    
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: int = 120,
    ) -> None:
        super().__init__(
            base_url=base_url or os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8010/v1"),
            model=model or os.getenv("LLAMA_MODEL", "llama-3.2-3b"),
            api_key=api_key,
            timeout=timeout
        )


class Mistral(LocalLLM):
    """Mistral model client with default configuration.
    
    Assumes Mistral server is running with:
        python -m llama_cpp.server \
          --model ./models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
          --n_gpu_layers -1 --n_ctx 2048 --n_batch 256 --offload_kqv true \
          --host 127.0.0.1 --port 8011
    """
    
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: int = 120,
    ) -> None:
        super().__init__(
            base_url=base_url or os.getenv("MISTRAL_BASE_URL", "http://127.0.0.1:8011/v1"),
            model=model or os.getenv("MISTRAL_MODEL", "mistral-7b-instruct"),
            api_key=api_key,
            timeout=timeout
        )


__all__ = ["LocalLLM", "Llama", "Mistral", "ChatResponse"]
