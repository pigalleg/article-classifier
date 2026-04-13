"""Reasoning-only LLM worker for RA selection."""

from __future__ import annotations

import json
import re
import time

from openai import OpenAI

from .llm_base import LLMBase


class LLMReasoner(LLMBase):
    def __init__(
        self,
        model=None,
        temperature=None,
        max_retries=None,
        timeout=None,
        requests_per_minute=None,
        base_url=None,
        api_key=None,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
            requests_per_minute=requests_per_minute,
            base_url=base_url,
            api_key=api_key,
            openai_client_cls=OpenAI,
        )

    def _build_reasoning_prompt(self, abstract, candidates) -> str:
        lines = []
        for candidate in candidates:
            ra_id = candidate.get("RA2025_ID") or candidate.get("RA2025") or candidate.get("RA2025_id") or candidate.get("id")
            question = candidate.get("Question") or candidate.get("Question_Cleaned") or candidate.get("Questions - long") or candidate.get("question") or ""
            id_label = str(ra_id) if ra_id is not None else "-"
            lines.append(f"- ID {id_label}: {question}")

        candidate_text = "\n".join(lines)
        return f"""
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

    def _parse_reasoning_response(self, reply: str):
        json_text = self._extract_json_from_reply(reply)
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            return None, f"⚠️ Could not parse LLM output: {reply}", None

        ra_id = data.get("RA2025_ID") or data.get("RA2025") or data.get("id")
        if isinstance(ra_id, str):
            m2 = re.search(r"(\d+)", ra_id)
            if m2:
                ra_id = m2.group(1)

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

    def classify_with_reasoning(self, abstract, candidates):
        if not abstract or not candidates:
            return None, "No data available", None

        prompt = self._build_reasoning_prompt(abstract, candidates)

        for attempt in range(1, self.max_retries + 1):
            try:
                self._apply_rate_limit()

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )

                self._last_request_time = time.time()

                message_content = getattr(response.choices[0].message, "content", None) if getattr(response, "choices", None) else None
                reply = str(message_content).strip() if message_content is not None else str(response)
                return self._parse_reasoning_response(reply)

            except Exception as e:
                if "insufficient_quota" in str(e):
                    return None, "[Quota exceeded — please add billing or wait for reset]", None
                if self._is_model_unavailable_error(e):
                    base_url = self._endpoint_for_logs()
                    raise RuntimeError(
                        f"Configured model '{self.model}' is unavailable on endpoint '{base_url}'. "
                        f"Original error: {e}"
                    ) from e

                print(f"⚠️ Attempt {attempt}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(5 * attempt)
                else:
                    return None, f"[Error calling LLM: {e}]", None
