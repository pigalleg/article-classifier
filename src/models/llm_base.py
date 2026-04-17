"""Shared LLM runtime plumbing for reasoning and affinity workers."""

from __future__ import annotations

import difflib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import yaml


def _load_reasoner_settings() -> tuple[dict, dict, dict]:
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    try:
        with open(cfg_path, "r") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        cfg = {}
    affinity_defaults = cfg.get("runtime", {}).get("affinity", {})
    reasoner_defaults = cfg.get("runtime", {}).get("llm_reasoner", {})
    model_defaults = cfg.get("models", {}).get("llm", {})
    return affinity_defaults, reasoner_defaults, model_defaults


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_str(value, default: str) -> str:
    if value is None:
        return str(default)
    s = str(value).strip()
    return s if s else str(default)


def _is_backend_cloud(value: str) -> bool:
    return _as_str(value, "local").lower() == "cloud"


def _default_openai_client_cls():
    from openai import OpenAI

    return OpenAI


class LLMBase:
    """Shared transport/runtime behavior for LLM-backed workers."""

    def __init__(
        self,
        model=None,
        temperature=None,
        max_retries=None,
        timeout=None,
        requests_per_minute=None,
        base_url=None,
        api_key=None,
        openai_client_cls=None,
    ):
        self._affinity_defaults, self._reasoner_defaults, self._model_defaults = _load_reasoner_settings()

        client_cls = openai_client_cls or _default_openai_client_cls()

        backend_env_var = self._model_defaults.get("backend_env_var")
        model_env_var = self._model_defaults.get("model_env_var")
        base_url_env_var = self._model_defaults.get("base_url_env_var")
        api_key_env_var = self._model_defaults.get("api_key_env_var")
        rpm_env_var = self._model_defaults.get("rpm_env_var")

        configured_backend = _as_str(self._model_defaults.get("default_backend"), "local")
        selected_backend = (os.getenv(str(backend_env_var)) if backend_env_var else None) or configured_backend
        use_cloud_backend = _is_backend_cloud(selected_backend)
        active_profile = self._model_defaults.get("cloud", {}) if use_cloud_backend else self._model_defaults.get("local", {})

        effective_base_url_env_var = active_profile.get("base_url_env_var") or active_profile.get("base_url_env") or base_url_env_var
        effective_api_key_env_var = active_profile.get("api_key_env_var") or active_profile.get("api_key_env") or api_key_env_var

        env_base_url = os.getenv(str(effective_base_url_env_var)) if effective_base_url_env_var else None
        profile_base_url = active_profile.get("base_url")
        resolved_base_url = base_url if base_url is not None else (env_base_url if env_base_url is not None else profile_base_url)
        if isinstance(resolved_base_url, str):
            resolved_base_url = resolved_base_url.strip() or None
        self._effective_base_url_env_var = str(effective_base_url_env_var) if effective_base_url_env_var else None
        self._resolved_base_url = resolved_base_url

        profile_api_key = active_profile.get("api_key")
        env_api_key = os.getenv(str(effective_api_key_env_var)) if effective_api_key_env_var else None
        resolved_api_key = api_key if api_key is not None else (env_api_key if env_api_key is not None else profile_api_key)

        timeout = _as_int(timeout, _as_int(self._reasoner_defaults.get("timeout_seconds"), 60))

        if resolved_base_url:
            self.client = client_cls(base_url=resolved_base_url, api_key=resolved_api_key, timeout=timeout)
        else:
            self.client = client_cls(api_key=resolved_api_key, timeout=timeout)
        print(f"Using base URL: {resolved_base_url}")

        self.backend = "cloud" if use_cloud_backend else "local"

        profile_model = active_profile.get("model")
        env_model = os.getenv(str(model_env_var)) if model_env_var else None
        selected_model = model or env_model or profile_model or self._model_defaults.get("model")
        if selected_model is None:
            raise ValueError("Missing model configuration: set models.llm.model_env_var or models.llm.<backend>.model in settings.yaml")
        self.model: str = str(selected_model)
        print(f"Using backend: {self.backend}")
        print(f"Using model: {self.model}")
        self.temperature = _as_float(temperature, _as_float(self._reasoner_defaults.get("temperature"), 0.3))
        self.max_retries = _as_int(max_retries, _as_int(self._reasoner_defaults.get("max_retries"), 3))

        rpm_env = os.getenv(str(rpm_env_var)) if rpm_env_var else None
        rpm_default_reasoner = self._reasoner_defaults.get(
            "requests_per_minute_default_cloud" if use_cloud_backend else "requests_per_minute_default_local"
        )
        rpm_default_profile = active_profile.get("requests_per_minute_default") or rpm_default_reasoner
        rpm_candidate = requests_per_minute if requests_per_minute is not None else (rpm_env if rpm_env is not None else rpm_default_profile)
        rpm_val = _as_int(rpm_candidate, _as_int(rpm_default_reasoner, 1))
        self.requests_per_minute = max(0, rpm_val)
        self._min_interval = 60.0 / self.requests_per_minute if self.requests_per_minute > 0 else 0.0
        self._last_request_time = 0.0

    def _endpoint_for_logs(self) -> str:
        if self._resolved_base_url:
            return str(self._resolved_base_url)
        if self._effective_base_url_env_var:
            env_val = os.getenv(self._effective_base_url_env_var)
            if env_val:
                return str(env_val)
        return "<openai-cloud>"

    @staticmethod
    def _is_model_unavailable_error(err: Exception) -> bool:
        msg = str(err).lower()
        model_markers = [
            "model",
            "not found",
            "does not exist",
            "unknown model",
            "invalid model",
            "no such model",
        ]
        return any(token in msg for token in model_markers)

    def _apply_rate_limit(self):
        if self._min_interval > 0:
            elapsed = time.time() - self._last_request_time
            if elapsed < self._min_interval:
                to_sleep = self._min_interval - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)

    def _extract_json_from_reply(self, reply: str) -> str:
        json_text = None
        first = reply.find("{")
        last = reply.rfind("}")

        if first != -1 and last != -1 and last > first:
            json_text = reply[first:last + 1]
        else:
            mf = re.search(r"```(?:json)?\s*(.*?)\s*```", reply, re.DOTALL | re.IGNORECASE)
            if mf:
                content = mf.group(1).strip()
                f = content.find("{")
                l = content.rfind("}")
                if f != -1 and l != -1 and l > f:
                    json_text = content[f:l + 1]

        return json_text if json_text is not None else reply

    @staticmethod
    def _normalize_target_id(value) -> str:
        s = str(value).strip()
        m = re.fullmatch(r"-?\d+(?:\.\d+)?", s)
        if m:
            try:
                f = float(s)
                if f.is_integer():
                    return str(int(f))
            except Exception:
                pass
        return s

    @staticmethod
    def _compact_text(value: Any, max_len: int = 600) -> str:
        txt = re.sub(r"\s+", " ", str(value or "")).strip()
        return txt[:max_len]

    @staticmethod
    def _compact_reason(value: Any, max_len: int = 350) -> str | None:
        txt = re.sub(r"\s+", " ", str(value or "")).strip()
        return txt[:max_len] if txt else None

    @staticmethod
    def _fuzzy_match_key(key: str, candidate_ids: list[str]) -> str | None:
        matches = difflib.get_close_matches(key, candidate_ids, n=1, cutoff=0.6)
        return matches[0] if matches else None

    @staticmethod
    def _coerce_score(value: Any) -> float | None:
        try:
            if value is None:
                raise TypeError("Missing score")
            score = float(value)
            return max(0.0, min(100.0, score))
        except Exception:
            return None

    def _extract_affinity_value(self, value: Any) -> tuple[float | None, str | None]:
        if isinstance(value, dict):
            score_raw = value.get("score", value.get("affinity", value.get("value")))
            reason = self._compact_reason(value.get("reason", value.get("rationale", value.get("explanation"))))
        else:
            score_raw = value
            reason = None
        return self._coerce_score(score_raw), reason

    @staticmethod
    def _chunk_targets(targets: list, chunk_size: int) -> list:
        if chunk_size <= 0:
            return [targets]
        return [targets[i:i + chunk_size] for i in range(0, len(targets), chunk_size)]

    def _normalize_batch_scores(self, data: dict, targets: list, enable_reasons: bool | None = None) -> dict:
        out: dict[str, Any] = {}
        id_to_raw = {self._normalize_target_id(t.get("id")): t.get("id") for t in targets}
        use_reasons = getattr(self, "enable_affinity_reasons", False) if enable_reasons is None else bool(enable_reasons)

        for raw_key, raw_score in data.items():
            key = str(raw_key).strip()
            normalized_key = self._normalize_target_id(key)
            mapped_key = None

            if normalized_key in id_to_raw:
                mapped_key = self._normalize_target_id(id_to_raw[normalized_key])
            else:
                m = re.fullmatch(r"(?:RA|PRP)?\s*(\d+)", key, flags=re.IGNORECASE)
                if m:
                    idx = int(m.group(1)) - 1
                    if 0 <= idx < len(targets):
                        mapped_key = self._normalize_target_id(targets[idx].get("id"))

                if mapped_key is None:
                    key_lower = key.lower()
                    for norm_id, raw_id in id_to_raw.items():
                        raw_lower = str(raw_id).lower()
                        if key_lower in raw_lower or raw_lower in key_lower:
                            mapped_key = norm_id
                            break

                if mapped_key is None:
                    targets_ids = list(id_to_raw.values())
                    matches = self._fuzzy_match_key(key, targets_ids)
                    if matches:
                        mapped_key = self._normalize_target_id(id_to_raw[self._normalize_target_id(matches)])

            if mapped_key is None:
                continue

            if isinstance(raw_score, dict):
                score_raw = raw_score.get("score", raw_score.get("affinity", raw_score.get("value")))
                reason_raw = raw_score.get("reason", raw_score.get("rationale", raw_score.get("explanation")))
                score = self._coerce_score(score_raw)
                if use_reasons:
                    out[mapped_key] = {
                        "score": score,
                        "reason": self._compact_reason(reason_raw, max_len=350) if reason_raw is not None else None,
                    }
                else:
                    out[mapped_key] = score
                continue

            score = self._coerce_score(raw_score)
            if use_reasons:
                out[mapped_key] = {
                    "score": score,
                    "reason": None,
                }
            else:
                out[mapped_key] = score

        return out

    def _call_json_prompt_with_retries(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        debug: bool = False,
        log_tag: str = "llm",
    ) -> dict | None:
        for attempt in range(1, self.max_retries + 1):
            try:
                self._apply_rate_limit()
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                self._last_request_time = time.time()

                message_content = getattr(resp.choices[0].message, "content", None) if getattr(resp, "choices", None) else None
                reply = str(message_content).strip() if message_content is not None else str(resp)

                if debug:
                    print(f"[{log_tag}] raw reply: {reply}")

                json_text = self._extract_json_from_reply(reply)
                try:
                    data = json.loads(json_text)
                    return data if isinstance(data, dict) else None
                except json.JSONDecodeError as je:
                    if debug:
                        print(f"[{log_tag}] JSON parse failed: {je}")
                        print(f"[{log_tag}] raw content: {json_text}")
                    if attempt < self.max_retries:
                        time.sleep(2 * attempt)
                    else:
                        return None

            except Exception as e:
                if self._is_model_unavailable_error(e):
                    base_url = self._endpoint_for_logs()
                    raise RuntimeError(
                        f"Configured model '{self.model}' is unavailable on endpoint '{base_url}'. "
                        f"Original error: {e}"
                    ) from e

                if debug:
                    print(f"[{log_tag}] attempt {attempt} failed: {e}")

                if attempt < self.max_retries:
                    time.sleep(2 * attempt)
                else:
                    return None

        return None