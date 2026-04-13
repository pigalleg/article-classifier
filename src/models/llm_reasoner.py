"""Module: llm_reasoner.py
LLM agent that chooses the best RA2025 question among top candidates,
and provides an explanation of the reasoning.
"""

from openai import OpenAI
import json
import time
import re
import os
from datetime import datetime
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
        backend_env_var = model_defaults.get("backend_env_var")
        model_env_var = model_defaults.get("model_env_var")
        base_url_env_var = model_defaults.get("base_url_env_var")
        api_key_env_var = model_defaults.get("api_key_env_var")
        rpm_env_var = model_defaults.get("rpm_env_var")

        configured_backend = _as_str(model_defaults.get("default_backend"), "local")
        selected_backend = (os.getenv(str(backend_env_var)) if backend_env_var else None) or configured_backend
        use_cloud_backend = _is_backend_cloud(selected_backend)
        active_profile = model_defaults.get("cloud", {}) if use_cloud_backend else model_defaults.get("local", {})

        # Resolve env-var names with profile override first, then llm-global names.
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

        timeout = _as_int(timeout, _as_int(reasoner_defaults.get("timeout_seconds"), 60))

        # Build client (OpenAI SDK is compatible with base_url + api_key)
        if resolved_base_url:
            self.client = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key, timeout=timeout)
        else:
            self.client = OpenAI(api_key=resolved_api_key, timeout=timeout)
        print(f"Using base URL: {resolved_base_url}")
        self.backend = "cloud" if use_cloud_backend else "local"

        # Model and runtime params
        profile_model = active_profile.get("model")
        env_model = os.getenv(str(model_env_var)) if model_env_var else None
        self.model = (
            model
            or env_model
            or profile_model
            or model_defaults.get("model")
        )
        if self.model is None:
            raise ValueError("Missing model configuration: set models.llm.model_env_var or models.llm.<backend>.model in settings.yaml")
        print(f"Using backend: {self.backend}")
        print(f"Using model: {self.model}")
        self.temperature = _as_float(temperature, _as_float(reasoner_defaults.get("temperature"), 0.3))
        self.max_retries = _as_int(max_retries, _as_int(reasoner_defaults.get("max_retries"), 3))

        # Rate limiting (requests per minute)
        rpm_env = os.getenv(str(rpm_env_var)) if rpm_env_var else None
        rpm_default_reasoner = reasoner_defaults.get(
            "requests_per_minute_default_cloud" if use_cloud_backend else "requests_per_minute_default_local"
        )
        rpm_default_profile = active_profile.get("requests_per_minute_default") or rpm_default_reasoner
        rpm_candidate = requests_per_minute if requests_per_minute is not None else (rpm_env if rpm_env is not None else rpm_default_profile)
        rpm_val = _as_int(rpm_candidate, _as_int(rpm_default_reasoner, 1))
        # print(f"Requests per minute: {rpm_val}")
        self.requests_per_minute = max(0, rpm_val)
        self._min_interval = 60.0 / self.requests_per_minute if self.requests_per_minute > 0 else 0.0
        self._last_request_time = 0.0

        # Debug/verbosity flags
        debug_env = os.getenv("AFFINITY_DEBUG")
        self.affinity_debug = _as_bool(debug_env, _as_bool(affinity_defaults.get("affinity_debug"), False))

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

    def _build_affinity_prompt(
        self,
        abstract: str,
        targets_text: str,
        target_type: str,
        required_ids: list[str],
    ) -> str:
        """Build prompt for affinity evaluation."""
        required_ids_text = ", ".join(required_ids)
        if target_type == "PRP":
            return (
                "Evaluate PRP affinities for the abstract. "
                "Score the abstract’s relevance to the program from 0–100 using these ranges: 0–40 = Low or no relevance (topics may be tangentially related but do not directly address the program’s goals); 41–70 = Moderate relevance (clear connection, but not central—e.g., focuses on methods or secondary aspects rather than the program’s core problem); 71–100 = High relevance (directly and substantially addresses the program’s main objectives).\n"
                "Return ONLY one JSON object (no extra text).\n"
                "Rules:\n"
                "- Include ALL required IDs exactly once as keys; do not add/rename keys.\n"
                "- Values must be numbers in [0,100].\n"
                f"Required IDs: {required_ids_text}\n"
                f"Abstract:\n{abstract}\n\n"
                f"Primary Research Programmes:\n{targets_text}\n\n"
                "Output JSON example:\n"
                "{\"<ID_FROM_INPUT_1>\": 85, \"<ID_FROM_INPUT_2>\": 42}\n\n"
            )
        else:  # RA type
            return (
                "Evaluate RA question affinities for the abstract. "
                "Score the abstract’s relevance to the research question from 0–100 using these ranges: 0–40 = Low or no relevance (topics may be tangentially related but do not directly address the questions’s goals); 41–70 = Moderate relevance (clear connection, but not central—e.g., focuses on methods or secondary aspects rather than the questions’s core problem); 71–100 = High relevance (directly and substantially addresses the questions’s main objectives).\n"
                "Return ONLY one JSON object (no extra text).\n"
                "Rules:\n"
                "- Include ALL required IDs exactly once as keys; do not add/rename keys.\n"
                "- Values must be numbers in [0,100].\n"
                f"Required IDs: {required_ids_text}\n"
                f"Abstract:\n{abstract}\n\n"
                f"Research questions:\n{targets_text}\n\n"
                "Output JSON example:\n"
                "{\"<ID_FROM_INPUT_1>\": 85, \"<ID_FROM_INPUT_2>\": 42}\n\n"
            )

    def _build_prp_scope_prompt(
        self,
        abstract: str,
        targets_text: str,
        general_prp_description: str,
        required_ids: list[str],
    ) -> str:
        """Build a PRP prompt that includes explicit in-scope/out-of-scope metadata."""
        required_ids_text = ", ".join(required_ids)
        return (
            "Evaluate PRP scope and PRP affinities for the abstract. "
            "Score the abstract’s relevance to the program from 0–100 using these ranges: 0–40 = Low or no relevance (topics may be tangentially related but do not directly address the program’s goals); 41–70 = Moderate relevance (clear connection, but not central—e.g., focuses on methods or secondary aspects rather than the program’s core problem); 71–100 = High relevance (directly and substantially addresses the program’s main objectives).\n"
            "An abstract can be outside PRP scope.\n\n"
            "Return ONLY one JSON object (no extra text) with this schema:\n"
            "{\n"
            "  \"belongs_any_prp\": true,\n"
            "  \"membership_confidence\": 0,\n"
            "  \"affinity_scores\": {\n"
            "    \"<ID_FROM_INPUT_1>\": 0,\n"
            "    \"<ID_FROM_INPUT_2>\": 0\n"
            "  }\n"
            "}\n\n"
            "Rules:\n"
            "- belongs_any_prp: boolean.\n"
            "- membership_confidence: number in [0,100], and >0 if belongs_any_prp=true.\n"
            "- affinity_scores: include ALL required IDs exactly once; do not add or rename keys.\n"
            "- affinity_scores values: numbers in [0,100].\n"
            f"Required IDs: {required_ids_text}\n\n"
            f"Abstract:\n{abstract}\n\n"
            f"General PRP description (scope anchor):\n{general_prp_description}\n\n"
            f"Primary Research Programmes:\n{targets_text}\n\n"
            
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

        normalized_targets = [
            {
                "id": self._normalize_target_id(t.get("id")),
                "text": str(t.get("text", "")),
            }
            for t in targets
        ]
        targets_text = "\n".join(
            f"{i+1}. [{t['id']}] {t['text']}"
            for i, t in enumerate(normalized_targets)
        )
        required_ids = [t["id"] for t in normalized_targets]
        prompt = self._build_affinity_prompt(abstract, targets_text, target_type, required_ids)
        if self.affinity_debug:
            print(f"[rate_affinity_batch] prompt:\n{prompt}")
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
                    base_url = self._endpoint_for_logs()
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

    def parse_prp_batch_response(self, data: dict, targets: list) -> dict:
        """Parse and validate strict PRP scope response payload."""
        if not isinstance(data, dict):
            raise ValueError("PRP scope response must be a JSON object")

        if "belongs_any_prp" not in data or not isinstance(data["belongs_any_prp"], bool):
            raise ValueError("Missing or invalid 'belongs_any_prp' (bool required)")

        if "membership_confidence" not in data:
            raise ValueError("Missing 'membership_confidence'")
        try:
            membership_confidence = float(data["membership_confidence"])
        except Exception as e:
            raise ValueError("Invalid 'membership_confidence' (numeric required)") from e
        membership_confidence = max(0.0, min(100.0, membership_confidence))

        affinity_scores_raw = data.get("affinity_scores")
        if not isinstance(affinity_scores_raw, dict):
            raise ValueError("Missing or invalid 'affinity_scores' (object required)")

        scores = self._normalize_batch_scores(affinity_scores_raw, targets)
        expected_ids = [self._normalize_target_id(t.get("id")) for t in targets]
        id_to_text = {
            self._normalize_target_id(t.get("id")): str(t.get("text", ""))
            for t in targets
        }

        missing_ids = [tid for tid in expected_ids if tid not in scores]
        if missing_ids:
            missing_questions = [{"id": tid, "text": id_to_text.get(tid, "")} for tid in missing_ids]
            raise ValueError(f"Missing affinity scores for targets: {missing_questions}")

        invalid_ids = [tid for tid, val in scores.items() if val is None]
        if invalid_ids:
            invalid_questions = [{"id": tid, "text": id_to_text.get(tid, "")} for tid in invalid_ids]
            raise ValueError(f"Invalid affinity values for targets: {invalid_questions}")

        return {
            "belongs_any_prp": data["belongs_any_prp"],
            "membership_confidence": membership_confidence,
            "scores": scores,
        }

    def _log_prp_scope_failure(
        self,
        abstract: str,
        prp_targets: list,
        reason: str,
        attempt: int,
        raw_content: str | None = None,
    ) -> None:
        """Append structured PRP scope parsing failures to a JSONL log for auditability."""
        log_path_raw = os.getenv("AFFINITY_PRP_SCOPE_FAILURE_LOG") or "data/results/prp_scope_failures.jsonl"
        log_path = Path(log_path_raw)
        if not log_path.is_absolute():
            log_path = Path.cwd() / log_path

        entry = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "model": self.model,
            "attempt": int(attempt),
            "reason": str(reason),
            "abstract_preview": str(abstract)[:600],
            "targets": [
                {
                    "id": self._normalize_target_id(t.get("id")),
                    "text": str(t.get("text", ""))[:300],
                }
                for t in prp_targets
            ],
            "raw_content_preview": None if raw_content is None else str(raw_content)[:1200],
        }

        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=True) + "\n")
        except Exception as log_err:
            if self.affinity_debug:
                print(f"[rate_prp_affinity_with_scope] failed to write failure log: {log_err}")

    def rate_prp_affinity_with_scope(self, abstract: str, prp_targets: list, general_prp_description: str) -> dict:
        """Rate PRP affinities with strict in-scope/out-of-scope metadata."""
        if not abstract:
            raise ValueError("Abstract text cannot be empty for PRP scope evaluation")
        if not prp_targets:
            raise ValueError("PRP targets cannot be empty for PRP scope evaluation")
        if not str(general_prp_description).strip():
            raise ValueError("General PRP description cannot be empty for PRP scope evaluation")

        normalized_targets = [
            {
                "id": self._normalize_target_id(t.get("id")),
                "text": str(t.get("text", "")),
            }
            for t in prp_targets
        ]
        targets_text = "\n".join(
            f"{i+1}. [{t['id']}] {t['text']}"
            for i, t in enumerate(normalized_targets)
        )
        required_ids = [t["id"] for t in normalized_targets]
        prompt = self._build_prp_scope_prompt(
            abstract,
            targets_text,
            str(general_prp_description).strip(),
            required_ids,
        )
        if self.affinity_debug:
            print(f"[rate_prp_affinity_with_scope] prompt:\n{prompt}")
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
                    print(f"[rate_prp_affinity_with_scope] raw reply: {reply}")

                json_text = self._extract_json_from_reply(reply)
                try:
                    data = json.loads(json_text)
                    return self.parse_prp_batch_response(data, prp_targets)
                except (json.JSONDecodeError, ValueError) as e:
                    self._log_prp_scope_failure(
                        abstract=abstract,
                        prp_targets=prp_targets,
                        reason=str(e),
                        attempt=attempt,
                        raw_content=json_text,
                    )
                    if self.affinity_debug:
                        print(f"[rate_prp_affinity_with_scope] parse failed: {e}")
                        print(f"[rate_prp_affinity_with_scope] raw content: {json_text}")
                    if attempt < self.max_retries:
                        time.sleep(2 * attempt)
                    else:
                        raise ValueError(f"Failed to parse strict PRP scope response: {e}") from e

            except Exception as e:
                if self._is_model_unavailable_error(e):
                    base_url = self._endpoint_for_logs()
                    raise RuntimeError(
                        f"Configured model '{self.model}' is unavailable on endpoint '{base_url}'. "
                        f"Original error: {e}"
                    ) from e

                if attempt < self.max_retries:
                    if self.affinity_debug:
                        print(f"[rate_prp_affinity_with_scope] attempt {attempt} failed: {e}")
                    time.sleep(2 * attempt)
                else:
                    raise

        raise ValueError("PRP scope evaluation failed after retries")
