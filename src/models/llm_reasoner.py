"""Module: llm_reasoner.py
LLM agent that chooses the best RA2025 question among top candidates,
and provides an explanation of the reasoning.
"""

from openai import OpenAI
import json
import time
import re
import os
from pathlib import Path
import difflib

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

class LLMReasoner:
    """Uses an LLM (GPT) to select the best RA question among candidates.

    Adds simple client-side rate limiting to respect request quotas
    (requests per minute).
    """

    def __init__(self, model=None, temperature=None, max_retries=None, timeout=None,
                 requests_per_minute=None, base_url=None, api_key=None,
                 micro_batch_size=None):
        """
        If OPENAI_BASE_URL is set, connect to a local/OpenAI-compatible server (e.g., Ollama).
        Otherwise, use OpenAI cloud. API key is read from env if not provided.
        RPM: if not provided, use OPENAI_REQUESTS_PER_MINUTE; default higher for local.
        """
        affinity_defaults, reasoner_defaults, model_defaults = _load_reasoner_settings()

        # Resolve endpoint/backend and credentials from settings with env overrides.
        backend_env_var = _as_str(model_defaults.get("backend_env_var"), "LLM_BACKEND")
        model_env_var = _as_str(model_defaults.get("model_env_var"), "OPENAI_MODEL")
        base_url_env_var = _as_str(model_defaults.get("base_url_env_var"), "OPENAI_BASE_URL")
        api_key_env_var = _as_str(model_defaults.get("api_key_env_var"), "OPENAI_API_KEY")
        rpm_env_var = _as_str(model_defaults.get("rpm_env_var"), "OPENAI_REQUESTS_PER_MINUTE")

        configured_backend = _as_str(model_defaults.get("default_backend"), "local")
        selected_backend = os.getenv(backend_env_var) or configured_backend
        use_cloud_backend = _is_backend_cloud(selected_backend)
        active_profile = model_defaults.get("cloud", {}) if use_cloud_backend else model_defaults.get("local", {})

        env_base_url = os.getenv(base_url_env_var)
        profile_base_url = active_profile.get("base_url")
        resolved_base_url = base_url if base_url is not None else (env_base_url if env_base_url is not None else profile_base_url)
        if isinstance(resolved_base_url, str):
            resolved_base_url = resolved_base_url.strip() or None

        profile_api_key = active_profile.get("api_key")
        profile_api_key_env = active_profile.get("api_key_env")
        env_api_key = os.getenv(api_key_env_var)
        if env_api_key is None and profile_api_key_env:
            env_api_key = os.getenv(str(profile_api_key_env))
        resolved_api_key = api_key if api_key is not None else (env_api_key if env_api_key is not None else profile_api_key)

        timeout = _as_int(timeout, _as_int(reasoner_defaults.get("timeout_seconds"), 60))

        # Build client (OpenAI SDK is compatible with base_url + api_key)
        if resolved_base_url:
            self.client = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key, timeout=timeout)
        else:
            self.client = OpenAI(api_key=resolved_api_key, timeout=timeout)

        self.backend = "cloud" if use_cloud_backend else "local"

        # Model and runtime params
        profile_model = active_profile.get("model")
        self.model = (
            model
            or os.getenv(model_env_var)
            or profile_model
            or model_defaults.get("model")
            or "gpt-4o-mini"
        )
        print(f"Using backend: {self.backend}")
        print(f"Using model: {self.model}")
        self.temperature = _as_float(temperature, _as_float(reasoner_defaults.get("temperature"), 0.3))
        self.max_retries = _as_int(max_retries, _as_int(reasoner_defaults.get("max_retries"), 3))

        # Rate limiting (requests per minute)
        rpm_env = os.getenv(rpm_env_var)
        rpm_default_cloud = _as_int(reasoner_defaults.get("requests_per_minute_default_cloud"), 3)
        rpm_default_local = _as_int(reasoner_defaults.get("requests_per_minute_default_local"), 9999)
        rpm_default_profile = _as_int(
            active_profile.get("requests_per_minute_default"),
            rpm_default_cloud if use_cloud_backend else rpm_default_local,
        )
        try:
            rpm_val = int(
                requests_per_minute
                if requests_per_minute is not None
                else (
                    rpm_env
                    if rpm_env is not None
                    else rpm_default_profile
                )
            )
        except Exception:
            rpm_val = rpm_default_profile
        print(f"Requests per minute: {rpm_val}")
        self.requests_per_minute = max(0, rpm_val)
        self._min_interval = 60.0 / self.requests_per_minute if self.requests_per_minute > 0 else 0.0
        self._last_request_time = 0.0

        # Debug/verbosity flags
        debug_env = os.getenv("AFFINITY_DEBUG")
        self.affinity_debug = _as_bool(debug_env, _as_bool(affinity_defaults.get("affinity_debug"), False))

        print_mbs_env = os.getenv("AFFINITY_PRINT_MICRO_BATCH_SIZE")
        self.print_micro_batch_size_on_startup = _as_bool(
            print_mbs_env,
            _as_bool(affinity_defaults.get("print_micro_batch_size_on_startup"), False),
        )

        # Micro-batching for affinity evaluation (targets per API call)
        # Env var: AFFINITY_MICRO_BATCH_SIZE (default: 10)
        mbs_env = os.getenv("AFFINITY_MICRO_BATCH_SIZE")
        mbs_default = _as_int(reasoner_defaults.get("micro_batch_size"), 10)
        try:
            mbs_val = int(
                micro_batch_size
                if micro_batch_size is not None
                else (mbs_env if mbs_env is not None else mbs_default)
            )
        except Exception:
            mbs_val = mbs_default
        self.micro_batch_size = max(1, mbs_val)
        if self.print_micro_batch_size_on_startup:
            print(f"Micro-batch size: {self.micro_batch_size}")

    @staticmethod
    def _is_model_unavailable_error(err: Exception) -> bool:
        """Best-effort detection for unavailable/unknown model errors."""
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

    def classify_with_reasoning(self, abstract, candidates):
        """
        Given an abstract and a list of top-k candidate RA questions,
        ask the LLM to select the most appropriate one.
        Returns (RA2025_ID, reasoning explanation, confidence).
        """
        if not abstract or not candidates:
            return None, "No data available", None

        # Format candidate list for readability
        lines = []
        for c in candidates:
            # support several possible keys from different upstreams
            ra_id = c.get("RA2025_ID") or c.get("RA2025") or c.get("RA2025_id") or c.get("id")
            question = c.get("Question") or c.get("Question_Cleaned") or c.get("Questions - long") or c.get("question") or ""
            id_label = str(ra_id) if ra_id is not None else "-"
            lines.append(f"- ID {id_label}: {question}")

        candidate_text = "\n".join(lines)

        prompt = f"""
You are classifying a power systems research paper into the most relevant
Research Agenda 2025 (RA2025) question.

Abstract:
{abstract}

Candidate RA2025 questions:
{candidate_text}

Task:
1. Select the most relevant RA2025_ID from the list.
2. Explain briefly (2–3 sentences) why this question matches the abstract.
3. Provide a Confidence score in [0, 1] for how well the abstract matches the selected question.
Return your answer in strict JSON format:
{{"RA2025_ID": "<selected ID>", "Reason": "<short explanation>", "Confidence": <number between 0 and 1>}}
"""

        # Retry logic for timeouts or API errors
        for attempt in range(1, self.max_retries + 1):
            try:
                # Enforce client-side rate limit
                if self._min_interval > 0:
                    elapsed = time.time() - self._last_request_time
                    if elapsed < self._min_interval:
                        to_sleep = self._min_interval - elapsed
                        # ensure we don't sleep negative
                        if to_sleep > 0:
                            time.sleep(to_sleep)

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )

                # record last request time after successful call
                self._last_request_time = time.time()

                # Extract reply text depending on SDK shape
                try:
                    reply = response.choices[0].message.content.strip()
                except Exception:
                    # fallback for other response shapes
                    reply = str(response)

                # Try to extract the JSON object from the reply. Many LLMs
                # wrap JSON in markdown fences or add commentary, so we
                # extract the first {...} substring as a best-effort.
                json_text = None
                first = reply.find("{")
                last = reply.rfind("}")
                if first != -1 and last != -1 and last > first:
                    json_text = reply[first:last+1]
                else:
                    # try to strip code fences like ```json ... ```
                    m = re.search(r"```(?:json)?\s*(.*?)\s*```", reply, re.DOTALL | re.IGNORECASE)
                    if m:
                        content = m.group(1).strip()
                        f = content.find("{")
                        l = content.rfind("}")
                        if f != -1 and l != -1 and l > f:
                            json_text = content[f:l+1]
                        else:
                            json_text = content

                if json_text is None:
                    json_text = reply

                try:
                    data = json.loads(json_text)
                except json.JSONDecodeError:
                    # If parsing still fails, return a helpful message including
                    # the raw reply so the user can inspect the exact output.
                    return None, f"⚠️ Could not parse LLM output: {reply}", None

                # Normalize RA2025_ID if the model returned a label like "ID 38"
                ra_id = data.get("RA2025_ID") or data.get("RA2025") or data.get("id")
                if isinstance(ra_id, str):
                    m2 = re.search(r"(\d+)", ra_id)
                    if m2:
                        ra_id = m2.group(1)

                # Confidence parsing (optional); clamp to [0,1]
                conf = data.get("Confidence")
                try:
                    conf = float(conf)
                except Exception:
                    conf = None
                if conf is not None:
                    if conf < 0:
                        conf = 0.0
                    elif conf > 1:
                        conf = 1.0

                return ra_id, data.get("Reason"), conf

            except Exception as e:
                if "insufficient_quota" in str(e):
                    return None, "[Quota exceeded — please add billing or wait for reset]", None
                else:
                    print(f"⚠️ Attempt {attempt}/{self.max_retries} failed: {e}")
                    if attempt < self.max_retries:
                        time.sleep(5 * attempt)  # exponential backoff
                    else:
                        return None, f"[Error calling LLM: {e}]", None

    def rate_affinity(self, abstract: str, target_text: str, target_type: str = "RA"):
        """
        DEPRECATED: Use rate_affinity_batch() instead for better token efficiency.
        
        Return a numeric affinity (0–100) for the pair (abstract, target_text).
        target_type: "RA" or "PRP" (chooses prompt wording).
        Falls back to single-item batch call.
        """
        if not abstract or not target_text:
            return None
        
        # Convert to batch format and delegate
        targets = [{"id": "single", "text": target_text}]
        result = self.rate_affinity_batch(abstract, targets, target_type)
        return result.get("single")

    def _apply_rate_limit(self):
        """Apply client-side rate limiting."""
        if self._min_interval > 0:
            elapsed = time.time() - self._last_request_time
            if elapsed < self._min_interval:
                to_sleep = self._min_interval - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)

    def _extract_json_from_reply(self, reply: str) -> str:
        """Extract JSON object from reply, handling various formats."""
        json_text = None
        first = reply.find("{")
        last = reply.rfind("}")
        
        if first != -1 and last != -1 and last > first:
            json_text = reply[first:last+1]
        else:
            # Try removing code fences
            mf = re.search(r"```(?:json)?\s*(.*?)\s*```", reply, re.DOTALL | re.IGNORECASE)
            if mf:
                content = mf.group(1).strip()
                f = content.find("{")
                l = content.rfind("}")
                if f != -1 and l != -1 and l > f:
                    json_text = content[f:l+1]
        
        return json_text if json_text is not None else reply

    @staticmethod
    def _normalize_target_id(value) -> str:
        """Normalize target IDs so response keys map reliably to input IDs."""
        s = str(value).strip()
        # Normalize integer-like numeric ids, e.g. "38.0" -> "38"
        m = re.fullmatch(r"-?\d+(?:\.\d+)?", s)
        if m:
            try:
                f = float(s)
                if f.is_integer():
                    return str(int(f))
            except Exception:
                pass
        return s

    def _normalize_batch_scores(self, data: dict, targets: list) -> dict:
        """Normalize model response keys and map ordinal keys back to target IDs.
        Tries multiple strategies: exact match, ordinal, substring, fuzzy match.
        """
        out = {}
        id_to_raw = {
            self._normalize_target_id(t.get("id")): t.get("id")
            for t in targets
        }

        for raw_key, raw_score in data.items():
            key = str(raw_key).strip()
            normalized_key = self._normalize_target_id(key)

            mapped_key = None
            
            # Strategy 1: Exact normalized match
            if normalized_key in id_to_raw:
                mapped_key = self._normalize_target_id(id_to_raw[normalized_key])
            else:
                # Strategy 2: Ordinal outputs like RA1, RA2, PRP1, PRP2, 1, 2
                m = re.fullmatch(r"(?:RA|PRP)?\s*(\d+)", key, flags=re.IGNORECASE)
                if m:
                    idx = int(m.group(1)) - 1
                    if 0 <= idx < len(targets):
                        mapped_key = self._normalize_target_id(targets[idx].get("id"))
                
                # Strategy 3: Substring match (for abbreviated names like "Grid Flexibility" vs full name)
                if mapped_key is None:
                    key_lower = key.lower()
                    for norm_id, raw_id in id_to_raw.items():
                        raw_lower = str(raw_id).lower()
                        # Check if response key is contained in target ID or vice versa
                        if key_lower in raw_lower or raw_lower in key_lower:
                            mapped_key = norm_id
                            break
                
                # Strategy 4: Fuzzy match (for typos or slight variations)
                if mapped_key is None:
                    targets_ids = list(id_to_raw.values())
                    matches = difflib.get_close_matches(key, targets_ids, n=1, cutoff=0.6)
                    if matches:
                        mapped_key = self._normalize_target_id(id_to_raw[self._normalize_target_id(matches[0])])

            if mapped_key is None:
                continue

            try:
                val = float(raw_score)
                val = max(0.0, min(100.0, val))
                out[mapped_key] = val
            except (ValueError, TypeError):
                out[mapped_key] = None

        return out

    def _build_affinity_prompt(self, abstract: str, targets_text: str, target_type: str) -> str:
        """Build prompt for affinity evaluation."""
        if target_type == "PRP":
            return (
                "You are evaluating how strongly a research abstract relates to multiple Primary Research Programmes.\n"
                "For each programme below, rate the degree of relation on a scale from 0 to 100 "
                "(100 = totally related, 0 = not related at all).\n"
                "Return ONLY a JSON object with no extra text, using the EXACT bracketed IDs as keys.\n\n"
                f"Abstract:\n{abstract}\n\n"
                f"Primary Research Programmes:\n{targets_text}\n\n"
                "Return format example: {\"<ID_FROM_INPUT_1>\": 85, \"<ID_FROM_INPUT_2>\": 42}"
            )
        else:  # RA type
            return (
                "You are evaluating how strongly a research abstract relates to multiple Research Agenda 2025 questions.\n"
                "For each question below, rate the degree of relation on a scale from 0 to 100 "
                "(100 = totally related, 0 = not related at all).\n"
                "Return ONLY a JSON object with no extra text, using the EXACT bracketed IDs as keys.\n\n"
                f"Abstract:\n{abstract}\n\n"
                f"Research Agenda Questions:\n{targets_text}\n\n"
                "Return format example: {\"<ID_FROM_INPUT_1>\": 85, \"<ID_FROM_INPUT_2>\": 42}"
            )

    @staticmethod
    def _chunk_targets(targets: list, chunk_size: int) -> list:
        """Split targets into fixed-size chunks for micro-batching."""
        if chunk_size <= 0:
            return [targets]
        return [targets[i:i + chunk_size] for i in range(0, len(targets), chunk_size)]

    def _rate_affinity_batch_single_call(self, abstract: str, targets: list, target_type: str) -> dict:
        """Execute one API call for a target chunk."""
        if not abstract or not targets:
            return {}

        targets_text = "\n".join(
            f"{i+1}. [{self._normalize_target_id(t['id'])}] {t['text']}"
            for i, t in enumerate(targets)
        )

        prompt = self._build_affinity_prompt(abstract, targets_text, target_type)

        for attempt in range(1, self.max_retries + 1):
            try:
                self._apply_rate_limit()

                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                self._last_request_time = time.time()

                try:
                    reply = resp.choices[0].message.content.strip()
                except Exception:
                    reply = str(resp)

                if self.affinity_debug:
                    print(f"[rate_affinity_batch] raw reply: {reply}")

                json_text = self._extract_json_from_reply(reply)

                try:
                    data = json.loads(json_text)
                    return self._normalize_batch_scores(data, targets)

                except json.JSONDecodeError as je:
                    if self.affinity_debug:
                        print(f"[rate_affinity_batch] JSON parse failed: {je}")
                        print(f"[rate_affinity_batch] raw content: {json_text}")

                    if attempt < self.max_retries:
                        time.sleep(2 * attempt)
                    else:
                        return {}

            except Exception as e:
                if self._is_model_unavailable_error(e):
                    base_url = os.getenv("OPENAI_BASE_URL") or "<openai-cloud>"
                    raise RuntimeError(
                        f"Configured model '{self.model}' is unavailable on endpoint '{base_url}'. "
                        f"Original error: {e}"
                    ) from e

                if self.affinity_debug:
                    print(f"[rate_affinity_batch] attempt {attempt} failed: {e}")

                if attempt < self.max_retries:
                    time.sleep(2 * attempt)
                else:
                    return {}

        return {}

    def rate_affinity_batch(self, abstract: str, targets: list, target_type: str = "RA") -> dict:
        """
        Dynamically evaluate affinity for multiple targets in a single LLM call.
        
        Args:
            abstract: The research abstract text
            targets: List of dicts with keys:
                - "id": unique identifier (e.g., "RA1", "PRP_Climate")
                - "text": the target text to evaluate
            target_type: "RA" or "PRP" for prompt context
        
        Returns:
            Dict mapping target_id -> affinity_score (0-100), or empty dict on failure
        """
        if not abstract or not targets:
            return {}
        
        chunks = self._chunk_targets(targets, self.micro_batch_size)
        combined = {}

        if self.affinity_debug:
            print(
                f"[rate_affinity_batch] targets={len(targets)}, "
                f"micro_batch_size={self.micro_batch_size}, chunks={len(chunks)}"
            )

        for chunk in chunks:
            partial = self._rate_affinity_batch_single_call(
                abstract=abstract,
                targets=chunk,
                target_type=target_type,
            )
            combined.update(partial)

        return combined
