"""Module: llm_reasoner.py
LLM agent that chooses the best RA2025 question among top candidates,
and provides an explanation of the reasoning.
"""

from openai import OpenAI
import json
import time
import re

class LLMReasoner:
    """Uses an LLM (GPT) to select the best RA question among candidates.

    Adds simple client-side rate limiting to respect request quotas
    (requests per minute).
    """

    def __init__(self, model="gpt-5", temperature=0.3, max_retries=3, timeout=60, requests_per_minute: int = 3):
        # Configure client with a timeout
        self.client = OpenAI(timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

        # Rate limiting (requests per minute)
        try:
            rpm = int(requests_per_minute)
        except Exception:
            rpm = 3
        self.requests_per_minute = max(0, rpm)
        self._min_interval = 60.0 / self.requests_per_minute if self.requests_per_minute > 0 else 0.0
        self._last_request_time = 0.0

    def classify_with_reasoning(self, abstract, candidates):
        """
        Given an abstract and a list of top-k candidate RA questions,
        ask the LLM to select the most appropriate one.
        Returns (RA2025_ID, reasoning explanation).
        """
        if not abstract or not candidates:
            return None, "No data available"

        # Format candidate list for readability
        candidate_text = "\n".join([
            f"- ID {c['RA2025_ID']}: {c['Question']}" for c in candidates
        ])

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
Return your answer in strict JSON format:
{{"RA2025_ID": "<selected ID>", "Reason": "<short explanation>"}}
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

                reply = response.choices[0].message.content.strip()

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
                    return None, f"⚠️ Could not parse LLM output: {reply}"

                # Normalize RA2025_ID if the model returned a label like "ID 38"
                ra_id = data.get("RA2025_ID")
                if isinstance(ra_id, str):
                    m2 = re.search(r"(\d+)", ra_id)
                    if m2:
                        ra_id = m2.group(1)

                return ra_id, data.get("Reason")

            # 👇 Here’s where your snippet goes
            except Exception as e:
                if "insufficient_quota" in str(e):
                    return None, "[Quota exceeded — please add billing or wait for reset]"
                else:
                    print(f"⚠️ Attempt {attempt}/{self.max_retries} failed: {e}")
                    if attempt < self.max_retries:
                        time.sleep(5 * attempt)  # exponential backoff
                    else:
                        return None, f"[Error calling LLM: {e}]"
