"""Module: llm_reasoner.py
LLM agent that chooses the best RA2025 question among top candidates,
and provides an explanation of the reasoning.
"""

from openai import OpenAI
import json
import time
import re
import os

class LLMReasoner:
    """Uses an LLM (GPT) to select the best RA question among candidates.

    Adds simple client-side rate limiting to respect request quotas
    (requests per minute).
    """

    def __init__(self, model="gpt-5", temperature=0.3, max_retries=3, timeout=60,
                 requests_per_minute=None, base_url=None, api_key=None):
        """
        If OPENAI_BASE_URL is set, connect to a local/OpenAI-compatible server (e.g., Ollama).
        Otherwise, use OpenAI cloud. API key is read from env if not provided.
        RPM: if not provided, use OPENAI_REQUESTS_PER_MINUTE; default higher for local.
        """
        # Resolve endpoint and credentials
        base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_API_BASE_URL")
        api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")

        # Build client (OpenAI SDK is compatible with base_url + api_key)
        if base_url:
            self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        else:
            self.client = OpenAI(api_key=api_key, timeout=timeout)

        # Model and runtime params
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.temperature = temperature
        self.max_retries = max_retries

        # Rate limiting (requests per minute)
        rpm_env = os.getenv("OPENAI_REQUESTS_PER_MINUTE") or os.getenv("AFFINITY_RPM")
        try:
            rpm_val = int(requests_per_minute if requests_per_minute is not None else (rpm_env if rpm_env is not None else (9999 if base_url else 3)))
        except Exception:
            rpm_val = 9999 if base_url else 3
        self.requests_per_minute = max(0, rpm_val)
        self._min_interval = 60.0 / self.requests_per_minute if self.requests_per_minute > 0 else 0.0
        self._last_request_time = 0.0

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

        # Format candidate list for readability, including programme context if available
        lines = []
        for c in candidates:
            # support several possible keys from different upstreams
            ra_id = c.get("RA2025_ID") or c.get("RA2025") or c.get("RA2025_id") or c.get("id")
            question = c.get("Question") or c.get("Question_Cleaned") or c.get("Questions - long") or c.get("question") or ""
            primary = c.get("Primary_Programme") or c.get("Predicted_Primary_Programme")
            secondary = c.get("Secondary_Programme") or c.get("Predicted_Secondary_Programme")
            prog_suffix = ""
            if primary or secondary:
                p = primary if primary else "-"
                s = secondary if secondary else "-"
                prog_suffix = f" [Primary: {p}; Secondary: {s}]"
            id_label = str(ra_id) if ra_id is not None else "-"
            lines.append(f"- ID {id_label}{prog_suffix}: {question}")

        candidate_text = "\n".join(lines)

        prompt = f"""
You are classifying a power systems research paper into the most relevant
Research Agenda 2025 (RA2025) question.

Abstract:
{abstract}

Candidate RA2025 questions:
{candidate_text}

Important context:
- Questions that share the same Primary or Secondary Research Programme are thematically related.
- When candidates are very similar, prefer the one whose programme alignment best matches the abstract's themes.

Task:
1. Select the most relevant RA2025_ID from the list.
2. Explain briefly (2–3 sentences) why this question matches the abstract, optionally referencing programme alignment.
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
        Return a numeric affinity (0–100) for the pair (abstract, target_text).
        target_type: "RA" or "PRP" (chooses prompt wording).
        Deterministic (temperature=0) and robust numeric parsing with retries and client-side rate limiting.
        """
        if not abstract or not target_text:
            return None

        if target_type == "PRP":
            prompt = (
                "You are evaluating how strongly a research abstract relates to a Primary Research Programme.\n"
                "The programme is described below.\n"
                "Rate the degree of relation on a scale from 0 to 100 (100 = totally related, 0 = not related at all).\n"
                "Return only the number.\n\n"
                f"Abstract:\n{abstract}\n\n"
                f"Primary Research Programme:\n{target_text}\n"
            )
        else:
            prompt = (
                "You are evaluating how strongly a research abstract relates to a Research Agenda 2025 question.\n"
                "Rate the degree of relation on a scale from 0 to 100 (100 = totally related, 0 = not related at all).\n"
                "Return only the number.\n\n"
                f"Abstract:\n{abstract}\n\n"
                f"Research Agenda Question:\n{target_text}\n"
            )

        for attempt in range(1, self.max_retries + 1):
            try:
                # rate limit
                if self._min_interval > 0:
                    elapsed = time.time() - self._last_request_time
                    if elapsed < self._min_interval:
                        to_sleep = self._min_interval - elapsed
                        if to_sleep > 0:
                            time.sleep(to_sleep)

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

                if os.getenv("AFFINITY_DEBUG") == "1":
                    print(f"[rate_affinity] raw reply: {reply}")

                # extract first number and clamp to [0,100]
                m = re.search(r"(-?\d+(\.\d+)?)", reply)
                if not m:
                    # try stripping code fences then search again
                    mf = re.search(r"```(?:json)?\s*(.*?)\s*```", reply, re.DOTALL | re.IGNORECASE)
                    content = mf.group(1) if mf else reply
                    m = re.search(r"(-?\d+(\.\d+)?)", content)

                if not m:
                    return None

                val = float(m.group(1))
                if val < 0:
                    val = 0.0
                if val > 100:
                    val = 100.0
                return float(val)

            except Exception as e:
                if self._is_model_unavailable_error(e):
                    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_API_BASE_URL") or "<openai-cloud>"
                    raise RuntimeError(
                        f"Configured model '{self.model}' is unavailable on endpoint '{base_url}'. "
                        "Set OPENAI_MODEL to an available model for this backend or switch endpoint. "
                        f"Original error: {e}"
                    ) from e
                if os.getenv("AFFINITY_DEBUG") == "1":
                    print(f"[rate_affinity] attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(2 * attempt)
                else:
                    return None
