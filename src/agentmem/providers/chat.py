"""Chat model adapter.

Point the OpenAI SDK at the OpenRouter base URL (or any OpenAI-compatible
endpoint) and capture token usage on every call: the evaluation harness needs
input/output tokens and cost per turn.

OpenRouter likes two identification headers; they are harmless elsewhere.
"""

from dataclasses import dataclass

from openai import OpenAI

from agentmem.settings import Settings, get_settings


@dataclass(frozen=True)
class ChatResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.settings.api_key  # touch to raise a clear pydantic error if broken
        self._client = OpenAI(
            base_url=self.settings.openrouter_base_url,
            api_key=self.settings.api_key or "unset",
            default_headers={
                "HTTP-Referer": "http://localhost:8730",
                "X-Title": "Acme Loan Copilot",
            },
        )
        self.calls = 0
        self.total_usage = {"prompt": 0, "completion": 0}

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_format: dict | None = None,
        model: str | None = None,
    ) -> ChatResult:
        """One chat call. messages use the OpenAI shape: {"role", "content"}.

        `model` overrides the default chat model for this call (used by the
        judge, which may be configured to a different family via
        MEM_JUDGE_MODEL).
        """
        chosen = model or self.settings.chat_model
        resp = self._client.chat.completions.create(
            model=chosen,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **({"response_format": response_format} if response_format else {}),
        )
        self.calls += 1
        usage = resp.usage
        if usage is not None:
            self.total_usage["prompt"] += usage.prompt_tokens or 0
            self.total_usage["completion"] += usage.completion_tokens or 0
        text = resp.choices[0].message.content or ""
        return ChatResult(
            text=text,
            model=resp.model or chosen,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
