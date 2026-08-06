#!/usr/bin/env python3
"""Validate configured benchmark models against their OpenAI-compatible endpoints.

Checks per model:
- API key presence from configured env var.
- Optional visibility from models.list().
- Lightweight chat.completions probe call.
- Token usage from probe response when available.

Usage:
  python scripts/util/validate_benchmark_models.py
  python scripts/util/validate_benchmark_models.py --include-disabled
  python scripts/util/validate_benchmark_models.py --names "gpt-5.4-mini,gemini-3.1-flash-lite"
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = REPO_ROOT / "src" / "config" / "settings.yaml"


@dataclass
class ModelProfile:
    name: str
    backend: str
    model: str
    base_url: str | None
    api_key_env: str | None
    api_key: str | None
    enabled: bool


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _load_settings(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_profiles(settings: dict[str, Any], include_disabled: bool) -> list[ModelProfile]:
    llm_cfg = settings.get("models", {}).get("llm", {})
    bench_cfg = llm_cfg.get("benchmark", {})
    cloud_profile = llm_cfg.get("cloud", {})
    local_profile = llm_cfg.get("local", {})

    profiles: list[ModelProfile] = []
    for item in bench_cfg.get("models", []) or []:
        enabled = _as_bool(item.get("enabled"), True)
        if not enabled and not include_disabled:
            continue

        backend = str(item.get("backend") or llm_cfg.get("default_backend") or "local").strip().lower()
        active = cloud_profile if backend == "cloud" else local_profile

        name = str(item.get("name") or item.get("model") or "").strip()
        model = str(item.get("model") or item.get("name") or active.get("model") or "").strip()
        if not name or not model:
            continue

        base_url = item.get("base_url")
        if base_url is None:
            base_url = active.get("base_url")
        if isinstance(base_url, str):
            base_url = base_url.strip() or None

        api_key_env = item.get("api_key_env")
        if api_key_env is None:
            api_key_env = active.get("api_key_env")
        if api_key_env is not None:
            api_key_env = str(api_key_env).strip() or None

        api_key = item.get("api_key")
        if api_key is None and api_key_env:
            api_key = os.getenv(api_key_env)
        if api_key is None:
            api_key = active.get("api_key")
        if isinstance(api_key, str):
            api_key = api_key.strip() or None

        profiles.append(
            ModelProfile(
                name=name,
                backend=backend,
                model=model,
                base_url=base_url,
                api_key_env=api_key_env,
                api_key=api_key,
                enabled=enabled,
            )
        )

    return profiles


def _probe_model(profile: ModelProfile, timeout_seconds: int) -> tuple[str, str, dict[str, Any]]:
    if not profile.api_key:
        return (
            "missing_api_key",
            "No API key found for this profile",
            {
                "api_key_env": profile.api_key_env,
                "base_url": profile.base_url,
            },
        )

    client_kwargs: dict[str, Any] = {
        "api_key": profile.api_key,
        "timeout": timeout_seconds,
    }
    if profile.base_url:
        client_kwargs["base_url"] = profile.base_url

    client = OpenAI(**client_kwargs)

    listed = None
    listed_count = None
    list_error = None
    try:
        models_page = client.models.list()
        data = getattr(models_page, "data", []) or []
        ids = [getattr(item, "id", None) for item in data]
        listed = profile.model in ids
        listed_count = len(ids)
    except Exception as exc:
        list_error = str(exc)

    probe_error = None
    usage_dict: dict[str, Any] = {}
    try:
        response = client.chat.completions.create(
            model=profile.model,
            messages=[{"role": "user", "content": "Reply with: ok"}],
            temperature=0.0,
            max_tokens=4,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
    except Exception as exc:
        probe_error = str(exc)

    details: dict[str, Any] = {
        "api_key_env": profile.api_key_env,
        "base_url": profile.base_url,
        "listed": listed,
        "listed_count": listed_count,
        "list_error": list_error,
        "probe_error": probe_error,
        "usage": usage_dict,
    }

    if probe_error is None:
        return "ok", "Probe call succeeded", details
    if listed is False:
        return "unavailable_model", "Model not listed and probe call failed", details
    return "probe_failed", "Probe call failed", details


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate configured benchmark models/endpoints")
    parser.add_argument("--include-disabled", action="store_true", help="Include disabled models from settings")
    parser.add_argument("--names", default="", help="Comma-separated profile names to validate")
    parser.add_argument("--timeout-seconds", type=int, default=30, help="HTTP timeout for checks")
    args = parser.parse_args()

    settings = _load_settings(SETTINGS_PATH)
    profiles = _resolve_profiles(settings, include_disabled=args.include_disabled)

    if args.names.strip():
        selected = {name.strip() for name in args.names.split(",") if name.strip()}
        profiles = [p for p in profiles if p.name in selected]

    if not profiles:
        print("No model profiles selected for validation")
        return 1

    failures = 0
    print(f"Validating {len(profiles)} model profile(s)...")
    for profile in profiles:
        status, message, details = _probe_model(profile, timeout_seconds=args.timeout_seconds)
        if status != "ok":
            failures += 1
        print(
            f"[{status}] name={profile.name} model={profile.model} "
            f"backend={profile.backend} enabled={profile.enabled}"
        )
        print(f"  message: {message}")
        print(f"  base_url: {details.get('base_url')}")
        print(f"  api_key_env: {details.get('api_key_env')}")
        if details.get("listed") is not None:
            print(f"  listed: {details.get('listed')} (catalog size={details.get('listed_count')})")
        if details.get("list_error"):
            print(f"  list_error: {details.get('list_error')}")
        if details.get("probe_error"):
            print(f"  probe_error: {details.get('probe_error')}")
        usage = details.get("usage") or {}
        if usage:
            print(
                "  usage: "
                f"prompt={usage.get('prompt_tokens')} "
                f"completion={usage.get('completion_tokens')} total={usage.get('total_tokens')}"
            )

    print(f"Done. failures={failures}, total={len(profiles)}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
