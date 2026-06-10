"""
LLM Client — multi-provider with streaming, retry, fallback.
Supports: OpenAI, Anthropic, DeepSeek, local (Ollama), OpenRouter.
"""
import asyncio
import json
import time
import logging
import os
from typing import Optional, Dict, Any, List, AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Provider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    CUSTOM = "custom"


@dataclass
class LLMResponse:
    """Unified response from any provider."""
    content: str = ""
    tool_calls: List[Dict] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    finish_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0
    cached: bool = False
    raw: Dict = field(default_factory=dict)


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: Provider = Provider.OPENAI
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    timeout: float = 120
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_provider: Optional[Provider] = None
    fallback_model: str = ""
    stream: bool = True


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests_per_minute: int = 60, tokens_per_minute: int = 100000):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self._request_times: List[float] = []
        self._token_counts: List[tuple] = []  # (timestamp, count)
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int = 0):
        """Wait until rate limit allows."""
        async with self._lock:
            now = time.time()
            cutoff = now - 60

            # Clean old entries
            self._request_times = [t for t in self._request_times if t > cutoff]
            self._token_counts = [(t, c) for t, c in self._token_counts if t > cutoff]

            # Check RPM
            while len(self._request_times) >= self.rpm:
                sleep_time = self._request_times[0] - cutoff + 0.1
                await asyncio.sleep(max(sleep_time, 0.1))
                now = time.time()
                cutoff = now - 60
                self._request_times = [t for t in self._request_times if t > cutoff]

            # Check TPM
            total_tokens = sum(c for _, c in self._token_counts)
            while total_tokens + estimated_tokens > self.tpm:
                sleep_time = self._token_counts[0][0] - cutoff + 0.1
                await asyncio.sleep(max(sleep_time, 0.1))
                now = time.time()
                cutoff = now - 60
                self._token_counts = [(t, c) for t, c in self._token_counts if t > cutoff]
                total_tokens = sum(c for _, c in self._token_counts)

            self._request_times.append(now)
            if estimated_tokens:
                self._token_counts.append((now, estimated_tokens))


class LLMClient:
    """
    Unified LLM client with:
    - Multi-provider support (OpenAI, Anthropic, DeepSeek, Ollama, OpenRouter)
    - Streaming responses
    - Automatic retry with exponential backoff
    - Provider fallback chain
    - Rate limiting
    - Response caching
    - Token counting
    - Tool/function calling
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.rate_limiter = RateLimiter()
        self._cache: Dict[str, LLMResponse] = {}
        self._cache_max = 100
        self._total_requests = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._http_client = None

    async def _get_client(self):
        """Get or create HTTP client."""
        if self._http_client is None:
            try:
                import httpx
                self._http_client = httpx.AsyncClient(timeout=self.config.timeout)
            except ImportError:
                import aiohttp
                self._http_client = aiohttp.ClientSession()
        return self._http_client

    async def chat(self, messages: List[Dict], tools: List[Dict] = None,
                   stream: bool = None, **kwargs) -> LLMResponse:
        """
        Send chat completion request.
        Auto-retries on failure, falls back to alternate provider.
        """
        stream = stream if stream is not None else self.config.stream
        cache_key = self._cache_key(messages, tools, **kwargs)

        # Check cache for non-streaming
        if not stream and cache_key in self._cache:
            cached = self._cache[cache_key]
            cached.cached = True
            return cached

        await self.rate_limiter.acquire(estimated_tokens=500)

        for attempt in range(self.config.max_retries):
            try:
                start = time.monotonic()
                response = await self._call_provider(
                    self.config.provider, self.config.model,
                    messages, tools=tools, stream=stream, **kwargs
                )
                response.latency_ms = (time.monotonic() - start) * 1000
                self._total_requests += 1
                self._total_tokens += response.input_tokens + response.output_tokens

                # Cache non-streaming
                if not stream:
                    self._cache[cache_key] = response
                    if len(self._cache) > self._cache_max:
                        oldest = next(iter(self._cache))
                        del self._cache[oldest]

                return response

            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{self.config.max_retries} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    # Try fallback provider
                    if self.config.fallback_provider:
                        logger.info(f"Falling back to {self.config.fallback_provider.value}")
                        return await self._call_provider(
                            self.config.fallback_provider,
                            self.config.fallback_model or self.config.model,
                            messages, tools=tools, stream=stream, **kwargs
                        )
                    raise

    async def chat_stream(self, messages: List[Dict], tools: List[Dict] = None,
                          **kwargs) -> AsyncIterator[str]:
        """Stream chat completion, yielding tokens."""
        response = await self.chat(messages, tools=tools, stream=True, **kwargs)

        if response.raw.get("_stream"):
            async for chunk in response.raw["_stream"]:
                yield chunk
        else:
            # Non-streaming fallback: yield word by word
            for word in response.content.split():
                yield word + " "

    async def _call_provider(self, provider: Provider, model: str,
                             messages: List[Dict], tools: List[Dict] = None,
                             stream: bool = False, **kwargs) -> LLMResponse:
        """Call specific provider."""
        if provider == Provider.OPENAI:
            return await self._call_openai(model, messages, tools, stream, **kwargs)
        elif provider == Provider.ANTHROPIC:
            return await self._call_anthropic(model, messages, tools, stream, **kwargs)
        elif provider == Provider.DEEPSEEK:
            return await self._call_deepseek(model, messages, tools, stream, **kwargs)
        elif provider == Provider.OPENROUTER:
            return await self._call_openrouter(model, messages, tools, stream, **kwargs)
        elif provider == Provider.OLLAMA:
            return await self._call_ollama(model, messages, tools, stream, **kwargs)
        elif provider == Provider.CUSTOM:
            return await self._call_custom(model, messages, tools, stream, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def _call_openai(self, model, messages, tools, stream, **kwargs):
        """Call OpenAI-compatible API."""
        client = await self._get_client()
        base_url = self.config.base_url or "https://api.openai.com/v1"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": self.config.top_p,
            "stream": stream,
        }
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]

        resp = await client.post(f"{base_url}/chat/completions", json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        msg = choice["message"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=msg.get("content", ""),
            tool_calls=msg.get("tool_calls", []),
            model=data.get("model", model),
            provider="openai",
            finish_reason=choice.get("finish_reason", ""),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            raw=data,
        )

    async def _call_anthropic(self, model, messages, tools, stream, **kwargs):
        """Call Anthropic API."""
        client = await self._get_client()
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        # Convert messages format
        system_msg = ""
        api_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                api_messages.append({"role": m["role"], "content": m["content"]})

        body = {
            "model": model,
            "messages": api_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        if system_msg:
            body["system"] = system_msg
        if tools:
            body["tools"] = tools

        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            json=body, headers=headers
        )
        resp.raise_for_status()
        data = resp.json()

        content = ""
        tool_calls = []
        for block in data.get("content", []):
            if block["type"] == "text":
                content += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append({
                    "id": block["id"],
                    "name": block["name"],
                    "arguments": block["input"],
                })

        usage = data.get("usage", {})
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            model=data.get("model", model),
            provider="anthropic",
            finish_reason=data.get("stop_reason", ""),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            raw=data,
        )

    async def _call_deepseek(self, model, messages, tools, stream, **kwargs):
        """Call DeepSeek API (OpenAI-compatible)."""
        self.config.base_url = self.config.base_url or "https://api.deepseek.com/v1"
        return await self._call_openai(model, messages, tools, stream, **kwargs)

    async def _call_openrouter(self, model, messages, tools, stream, **kwargs):
        """Call OpenRouter API."""
        self.config.base_url = "https://openrouter.ai/api/v1"
        return await self._call_openai(model, messages, tools, stream, **kwargs)

    async def _call_ollama(self, model, messages, tools, stream, **kwargs):
        """Call local Ollama API."""
        client = await self._get_client()
        base_url = self.config.base_url or "http://localhost:11434"
        body = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        resp = await client.post(f"{base_url}/api/chat", json=body)
        resp.raise_for_status()
        data = resp.json()

        msg = data.get("message", {})
        return LLMResponse(
            content=msg.get("content", ""),
            model=model,
            provider="ollama",
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            raw=data,
        )

    async def _call_custom(self, model, messages, tools, stream, **kwargs):
        """Call custom endpoint (OpenAI-compatible)."""
        return await self._call_openai(model, messages, tools, stream, **kwargs)

    def _cache_key(self, messages, tools, **kwargs):
        """Generate cache key."""
        import hashlib
        content = json.dumps({"m": messages, "t": tools, "k": kwargs}, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def stats(self):
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "cache_size": len(self._cache),
            "provider": self.config.provider.value,
            "model": self.config.model,
        }

    async def close(self):
        if self._http_client:
            await self._http_client.aclose()
