"""Adjudication LLM worker to consolidate divergent affinity signals across models."""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from .llm_base import LLMBase, _as_bool, _as_int


class LLMAffinityAdjudicatorReasoner(LLMBase):
    """LLM adjudicator for final affinity resolution using cross-model evidence."""

    def __init__(
        self,
        model=None,
        temperature=None,
        max_retries=None,
        timeout=None,
        requests_per_minute=None,
        base_url=None,
        api_key=None,
        micro_batch_size=None,
        enable_affinity_reasons=None,
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

        debug_env = os.getenv("AFFINITY_DEBUG")
        self.affinity_debug = _as_bool(debug_env, _as_bool(self._affinity_defaults.get("affinity_debug"), False))

        mbs_env = os.getenv("AFFINITY_MICRO_BATCH_SIZE")
        mbs_default = _as_int(self._reasoner_defaults.get("micro_batch_size"), 10)
        try:
            mbs_val = int(
                micro_batch_size
                if micro_batch_size is not None
                else (mbs_env if mbs_env is not None else mbs_default)
            )
        except Exception:
            mbs_val = mbs_default
        self.micro_batch_size = max(1, mbs_val)

        enable_affinity_reasons_env = os.getenv("AFFINITY_ENABLE_AFFINITY_REASONS")
        self.enable_affinity_reasons = _as_bool(
            enable_affinity_reasons if enable_affinity_reasons is not None else enable_affinity_reasons_env,
            _as_bool(self._reasoner_defaults.get("enable_affinity_reasons"), False),
        )

    def _build_adjudication_prompt(
        self,
        abstract: str,
        targets_text: str,
        target_type: str,
        required_ids: list[str],
    ) -> str:
        required_ids_text = ", ".join(required_ids)

        if self.enable_affinity_reasons:
            rules_block = (
                "- For each required ID, return an object with keys 'score' and 'reason'.\n"
                "- The reason must explain how conflicting model signals were resolved.\n"
            )
            example_block = (
                '{"<ID_FROM_INPUT_1>": {"score": 72, "reason": "conflict resolved using strongest evidence"}, '
                '"<ID_FROM_INPUT_2>": {"score": 35, "reason": "majority low affinity with consistent rationale"}}\n\n'
            )
        else:
            rules_block = "- Values must be numbers in [0,100].\n"
            example_block = '{"<ID_FROM_INPUT_1>": 72, "<ID_FROM_INPUT_2>": 35}\n\n'

        target_label = "Primary Research Programmes" if str(target_type).upper() == "PRP" else "Research questions"
        return (
            "You are an adjudicator consolidating affinity assessments from multiple models. "
            "For each target ID, synthesize the evidence and return a final affinity score from 0 to 100.\n"
            "Return ONLY one JSON object (no extra text).\n"
            "Rules:\n"
            "- Include ALL required IDs exactly once as keys; do not add/rename keys.\n"
            f"{rules_block}"
            f"Required IDs: {required_ids_text}\n\n"
            f"Abstract:\n{abstract}\n\n"
            f"{target_label} with model evidence:\n{targets_text}\n\n"
            "Output JSON example:\n"
            f"{example_block}"
        )

    def _format_target_evidence(self, targets: list[dict[str, Any]]) -> tuple[str, list[str]]:
        lines: list[str] = []
        required_ids: list[str] = []

        for idx, target in enumerate(targets, start=1):
            target_id = self._normalize_target_id(target.get("id"))
            target_text = str(target.get("text", "")).strip()
            evidence = target.get("model_evidence") or []

            required_ids.append(target_id)
            lines.append(f"{idx}. [{target_id}] {target_text}")

            if not isinstance(evidence, list) or not evidence:
                lines.append("   - No model evidence provided")
                continue

            for record in evidence:
                if not isinstance(record, dict):
                    continue
                model_slug = str(record.get("model_slug", "unknown-model")).strip() or "unknown-model"
                score = self._coerce_score(record.get("llm_affinity"))
                reason = self._compact_reason(record.get("llm_affinity_reason"), max_len=350)
                lines.append(f"   - Model={model_slug}; Affinity={score}")
                if reason:
                    lines.append(f"     Reason={reason}")

        return "\n".join(lines), required_ids

    def _adjudicate_batch_single_call(self, abstract: str, targets: list[dict[str, Any]], target_type: str) -> dict:
        if not abstract or not targets:
            return {}

        targets_text, required_ids = self._format_target_evidence(targets)
        prompt = self._build_adjudication_prompt(abstract, targets_text, target_type, required_ids)

        if self.affinity_debug:
            print(f"[adjudicate_batch] prompt:\n{prompt}")
        
        data = self._call_json_prompt_with_retries(
            prompt,
            temperature=0.0,
            debug=self.affinity_debug,
            log_tag="adjudicate_batch",
        )
        if not isinstance(data, dict):
            return {}

        normalize_targets = [{"id": t.get("id")} for t in targets]
        return self._normalize_batch_scores(data, normalize_targets, enable_reasons=self.enable_affinity_reasons)

    def adjudicate_batch(self, abstract: str, targets: list[dict[str, Any]], target_type: str = "PRP") -> dict:
        if not abstract or not targets:
            return {}

        chunks = self._chunk_targets(targets, self.micro_batch_size)
        combined: dict[str, Any] = {}

        if self.affinity_debug:
            print(
                f"[adjudicate_batch] targets={len(targets)}, "
                f"micro_batch_size={self.micro_batch_size}, chunks={len(chunks)}"
            )
        for chunk in chunks:
            partial = self._adjudicate_batch_single_call(
                abstract=abstract,
                targets=chunk,
                target_type=target_type,
            )
            combined.update(partial)

        return combined

    def adjudicate_one(self, abstract: str, target: dict[str, Any], target_type: str = "PRP") -> Any:
        if not abstract or not target:
            return None

        target_id = self._normalize_target_id(target.get("id"))
        out = self.adjudicate_batch(abstract, [target], target_type=target_type)
        return out.get(target_id)
