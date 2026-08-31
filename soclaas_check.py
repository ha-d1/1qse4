"""Safe SoCLaaS connectivity check: prints model IDs, never credentials."""
from __future__ import annotations

import argparse
import os

from soclaas_client import DEFAULT_BASE_URL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("SOCLAAS_MODEL", "qwen3-coder-next"))
    args = parser.parse_args()
    api_key = os.environ.get("SOCLAAS_API_KEY")
    if not api_key:
        raise SystemExit("SOCLAAS_API_KEY is not set")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install dependencies with: python3 -m pip install -r requirements.txt") from exc
    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("SOCLAAS_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
    )
    try:
        model_ids = sorted(model.id for model in client.models.list().data)
    finally:
        client.close()
    print(f"Connected. {len(model_ids)} model(s) available.")
    print(f"Requested model available: {'yes' if args.model in model_ids else 'no'}")
    for model_id in model_ids:
        print(model_id)


if __name__ == "__main__":
    main()
