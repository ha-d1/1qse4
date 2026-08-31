"""Conservative per-call reservations for bounded LLM iterations."""
from __future__ import annotations


class TokenBudgetExceeded(RuntimeError):
    """Raised before an API call whose worst-case reservation exceeds the budget."""


def conservative_prompt_tokens(*parts: str, overhead_tokens: int = 512) -> int:
    """Upper-bound byte-tokenized prompts without depending on a provider tokenizer.

    Qwen tokenization normally packs multiple UTF-8 bytes into a token. Counting every byte as
    one token is deliberately conservative and keeps the guard useful even when the remote
    tokenizer is unavailable locally. ``overhead_tokens`` covers chat framing and JSON mode.
    """
    if overhead_tokens < 0:
        raise ValueError("Prompt overhead must be non-negative")
    return overhead_tokens + sum(len(part.encode("utf-8")) for part in parts)


def reserve_output_tokens(
    *,
    token_cap: int,
    tokens_spent: int,
    requested_output_tokens: int,
    minimum_output_tokens: int,
    prompt_parts: tuple[str, ...],
    overhead_tokens: int = 512,
) -> dict[str, int]:
    """Return a safe output limit or stop locally before the provider call."""
    values = (token_cap, tokens_spent, requested_output_tokens, minimum_output_tokens)
    if any(value < 0 for value in values):
        raise ValueError("Token budget values must be non-negative")
    if token_cap == 0:
        return {
            "prompt_reservation": 0,
            "output_reservation": requested_output_tokens,
            "total_reservation": requested_output_tokens,
            "remaining_before_call": 0,
        }
    remaining = token_cap - tokens_spent
    prompt_reservation = conservative_prompt_tokens(
        *prompt_parts, overhead_tokens=overhead_tokens
    )
    available_output = remaining - prompt_reservation
    if available_output < minimum_output_tokens:
        raise TokenBudgetExceeded(
            "LLM call blocked before transmission: "
            f"remaining={remaining}, prompt_reservation={prompt_reservation}, "
            f"minimum_output={minimum_output_tokens}, token_cap={token_cap}"
        )
    output_reservation = min(requested_output_tokens, available_output)
    return {
        "prompt_reservation": prompt_reservation,
        "output_reservation": output_reservation,
        "total_reservation": prompt_reservation + output_reservation,
        "remaining_before_call": remaining,
    }
