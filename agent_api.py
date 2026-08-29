"""Minimal OpenAI-compatible chat-completions transport."""
import json
import os
import socket
import time
from urllib import error, parse, request


_MAX_RESPONSE_BYTES = 512 * 1024
_RETRY_DELAYS = (1, 2, 4)


class AgentAPIError(RuntimeError):
    pass


class OpenAICompatibleClient:
    def __init__(self, base_url, model, api_key, timeout_seconds=60):
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("API base URL is required")
        parsed = parse.urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("API base URL must be an absolute HTTP(S) URL")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("API model is required")
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("AGENT_API_KEY is required")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _request(self, messages, request_id):
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }, separators=(",", ":")).encode("utf-8")
        return request.Request(
            self.base_url + "/chat/completions",
            data=payload,
            method="POST",
            headers={
                "Authorization": "Bearer " + self._api_key,
                "Content-Type": "application/json",
                "X-Request-ID": request_id,
            },
        )

    @staticmethod
    def _read_response(response):
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                if int(content_length) > _MAX_RESPONSE_BYTES:
                    raise AgentAPIError("Provider response exceeds 512 KiB")
            except ValueError:
                pass
        body = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(body) > _MAX_RESPONSE_BYTES:
            raise AgentAPIError("Provider response exceeds 512 KiB")
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise AgentAPIError("Provider returned malformed JSON") from None

    @staticmethod
    def _parse_content(envelope):
        try:
            choices = envelope["choices"]
            if not isinstance(choices, list) or not choices:
                raise TypeError
            message = choices[0]["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError):
            raise AgentAPIError("Provider response is missing choices[0].message.content") from None
        if not isinstance(content, str):
            raise AgentAPIError("Provider choices[0].message.content must be a string")
        stripped = content.lstrip()
        try:
            value, end = json.JSONDecoder().raw_decode(stripped)
        except json.JSONDecodeError:
            raise AgentAPIError("Model content is not a JSON object") from None
        if stripped[end:].strip() or not isinstance(value, dict):
            raise AgentAPIError("Model content must contain exactly one JSON object")
        return value

    def complete(self, messages, request_id):
        for attempt in range(len(_RETRY_DELAYS) + 1):
            api_request = self._request(messages, request_id)
            try:
                with request.urlopen(api_request, timeout=self.timeout_seconds) as response:
                    envelope = self._read_response(response)
                return self._parse_content(envelope)
            except error.HTTPError as exc:
                status = exc.code
                exc.close()
                retryable = status == 429 or 500 <= status <= 599
                if not retryable:
                    raise AgentAPIError(f"Provider rejected request with HTTP {status}") from None
                failure = f"Provider request failed with retryable HTTP {status}"
            except (error.URLError, socket.timeout, TimeoutError):
                failure = "Provider network request failed"
            if attempt == len(_RETRY_DELAYS):
                raise AgentAPIError(failure + " after four attempts")
            time.sleep(_RETRY_DELAYS[attempt])
        raise AssertionError("unreachable")


def client_from_environment(api_base=None, api_model=None, timeout_seconds=60, environ=None):
    values = os.environ if environ is None else environ
    key = values.get("AGENT_API_KEY")
    base = api_base or values.get("AGENT_API_BASE")
    model = api_model or values.get("AGENT_MODEL")
    missing = [name for name, value in (("AGENT_API_KEY", key),
                                        ("AGENT_API_BASE", base),
                                        ("AGENT_MODEL", model)) if not value]
    if missing:
        raise ValueError("Missing required API configuration: " + ", ".join(missing))
    return OpenAICompatibleClient(base, model, key, timeout_seconds=timeout_seconds)
