"""Affinity scoring worker for RA and PRP evaluation."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

from .llm_base import LLMBase, _as_bool, _as_int


class LLMAffinityReasoner(LLMBase):
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
        enable_few_shot=None,
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

        enable_few_shot_env = os.getenv("AFFINITY_ENABLE_FEW_SHOT")
        self.enable_few_shot = _as_bool(
            enable_few_shot if enable_few_shot is not None else enable_few_shot_env,
            _as_bool(self._reasoner_defaults.get("enable_few_shot"), False),
        )
        enable_affinity_reasons_env = os.getenv("AFFINITY_ENABLE_AFFINITY_REASONS")
        self.enable_affinity_reasons = _as_bool(
            enable_affinity_reasons if enable_affinity_reasons is not None else enable_affinity_reasons_env,
            _as_bool(self._reasoner_defaults.get("enable_affinity_reasons"), False),
        )
        few_shot_max_env = os.getenv("AFFINITY_FEW_SHOT_MAX_EXAMPLES")
        few_shot_max_default = _as_int(self._reasoner_defaults.get("few_shot_max_examples_per_prompt"), 2)
        self.few_shot_max_examples_per_prompt = max(
            0,
            _as_int(
                few_shot_max_env if few_shot_max_env is not None else few_shot_max_default,
                few_shot_max_default,
            ),
        )

        self.few_shot_ra_examples: list[dict[str, Any]] = []
        self.few_shot_prp_examples: list[dict[str, Any]] = []
        if self.enable_few_shot:
            ra_path_raw = os.getenv("AFFINITY_FEW_SHOT_RA_FILE") or self._reasoner_defaults.get("few_shot_ra_file")
            prp_path_raw = os.getenv("AFFINITY_FEW_SHOT_PRP_FILE") or self._reasoner_defaults.get("few_shot_prp_file")
            self.few_shot_ra_examples = self._load_few_shot_examples(ra_path_raw, expected_target_type="RA")
            self.few_shot_prp_examples = self._load_few_shot_examples(prp_path_raw, expected_target_type="PRP")

    def _resolve_repo_path(self, path_raw: Any) -> Path | None:
        if path_raw is None:
            return None
        raw = str(path_raw).strip()
        if not raw:
            return None
        path = Path(raw).expanduser()
        if path.is_absolute():
            return path
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / path

    def _load_few_shot_examples(self, path_raw: Any, expected_target_type: str) -> list[dict[str, Any]]:
        expected = str(expected_target_type).strip().upper()
        path = self._resolve_repo_path(path_raw)
        if path is None:
            return []
        if not path.exists():
            print(f"Warning: few-shot file not found for {expected}: {path}")
            return []

        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            print(f"Warning: failed to load few-shot file for {expected}: {path} ({exc})")
            return []

        items = payload.get("examples") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            print(f"Warning: invalid few-shot format for {expected}: {path} (expected a list under 'examples')")
            return []

        out: list[dict[str, Any]] = []
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue

            item_type = str(item.get("target_type", expected)).strip().upper()
            if item_type != expected:
                continue

            abstract = self._compact_text(item.get("abstract"), max_len=900)
            if not abstract:
                continue

            labels: list[dict[str, Any]] = []

            targets_raw = item.get("targets")
            if isinstance(targets_raw, list):
                for target in targets_raw:
                    if not isinstance(target, dict):
                        continue
                    target_id = self._compact_text(target.get("target_id"), max_len=120)
                    if not target_id:
                        continue
                    score_raw = target.get("score")
                    if score_raw is None:
                        continue
                    try:
                        score = float(score_raw)
                    except Exception:
                        continue
                    labels.append(
                        {
                            "target_id": target_id,
                            "score": max(0.0, min(120.0, score)),
                            "rationale": self._compact_text(target.get("rationale"), max_len=300),
                        }
                    )

            if not labels:
                target_ids_raw = item.get("target_ids")
                scores_raw = item.get("scores")
                rationales_raw = item.get("rationales")
                if isinstance(target_ids_raw, list) and isinstance(scores_raw, list):
                    rationales_list = rationales_raw if isinstance(rationales_raw, list) else []
                    for i, target_id_raw in enumerate(target_ids_raw):
                        target_id = self._compact_text(target_id_raw, max_len=120)
                        if not target_id or i >= len(scores_raw):
                            continue
                        try:
                            score = float(scores_raw[i])
                        except Exception:
                            continue
                        rationale_raw = rationales_list[i] if i < len(rationales_list) else ""
                        labels.append(
                            {
                                "target_id": target_id,
                                "score": max(0.0, min(120.0, score)),
                                "rationale": self._compact_text(rationale_raw, max_len=300),
                            }
                        )

            if not labels:
                target_id = self._compact_text(item.get("target_id"), max_len=120)
                if target_id:
                    score_raw = item.get("score")
                    if score_raw is not None:
                        try:
                            score = float(score_raw)
                            labels.append(
                                {
                                    "target_id": target_id,
                                    "score": max(0.0, min(120.0, score)),
                                    "rationale": self._compact_text(item.get("rationale"), max_len=300),
                                }
                            )
                        except Exception:
                            pass

            if not labels:
                continue

            out.append(
                {
                    "id": self._compact_text(item.get("id") or f"{expected}_{idx}", max_len=80),
                    "target_type": expected,
                    "abstract": abstract,
                    "target_id": labels[0]["target_id"],
                    "score": labels[0]["score"],
                    "rationale": labels[0]["rationale"],
                    "targets": labels,
                }
            )

        return out

    def _format_few_shot_block(self, target_type: str) -> str:
        if not self.enable_few_shot:
            return ""

        src = self.few_shot_prp_examples if str(target_type).upper() == "PRP" else self.few_shot_ra_examples
        if not src:
            return ""

        selected = src if self.few_shot_max_examples_per_prompt <= 0 else src[: self.few_shot_max_examples_per_prompt]

        lines = ["Few-shot scoring examples for calibration references:"]
        for i, ex in enumerate(selected, start=1):
            lines.append(f"Example {i}:")
            lines.append(f"- Abstract: {ex['abstract']}")
            targets = ex.get("targets") if isinstance(ex.get("targets"), list) else []
            if targets:
                for j, target in enumerate(targets, start=1):
                    lines.append(f"- Label {j} Target ID: {target.get('target_id', '')}")
                    lines.append(f"- Label {j} Score: {float(target.get('score', 0.0)):.1f}")
                    rationale = self._compact_text(target.get("rationale"), max_len=300)
                    if rationale:
                        lines.append(f"- Label {j} Rationale: {rationale}")
            else:
                lines.append(f"- Target ID: {ex['target_id']}")
                lines.append(f"- Score: {float(ex['score']):.1f}")
                if ex.get("rationale"):
                    lines.append(f"- Rationale: {ex['rationale']}")
        lines.append("Use these only as scoring calibration references; evaluate the current abstract independently.")
        return "\n".join(lines) + "\n\n"

    def _build_affinity_prompt(
        self,
        abstract: str,
        targets_text: str,
        target_type: str,
        required_ids: list[str],
    ) -> str:
        required_ids_text = ", ".join(required_ids)
        if target_type == "PRP":
            few_shot_block = self._format_few_shot_block("PRP")
            if self.enable_affinity_reasons:
                rules_block = (
                    "- For each required ID, return an object with keys 'score' and 'reason'.\n"
                    "- The reason must be short and specific to that ID.\n"
                )
                example_block = (
                    '{"<ID_FROM_INPUT_1>": {"score": 85, "reason": "brief reason"}, "<ID_FROM_INPUT_2>": {"score": 42, "reason": "brief reason"}}\n\n'
                )
            else:
                rules_block = "- Values must be numbers in [0,120].\n"
                example_block = '{"<ID_FROM_INPUT_1>": 85, "<ID_FROM_INPUT_2>": 42}\n\n'
            return (
                "You are a power systems expert evaluator assessing research scope. "
                "Evaluate the affinity between the abstract and a set of Primary Research Programmes (PRP). "
                "Score the abstract’s relevance to the program from 0–120 using these ranges: 1–40 = Low relevance (0 = no relevance; topics may be tangentially related but do not directly address the program’s goals); 41–80 = Moderate relevance (clear connection, but not central—e.g., focuses on methods or secondary aspects rather than the program’s core problem); 81–120 = High relevance (directly and substantially addresses the program’s main objectives).\n"
                "Return ONLY one JSON object (no extra text).\n"
                "Rules:\n"
                "- Include ALL required IDs exactly once as keys; do not add/rename keys.\n"
                f"{rules_block}"
                f"Required IDs: {required_ids_text}\n"
                f"{few_shot_block}"
                f"Abstract:\n{abstract}\n\n"
                f"Primary Research Programmes:\n{targets_text}\n\n"
                "Output JSON example:\n"
                f"{example_block}"
            )

        few_shot_block = self._format_few_shot_block("RA")
        if self.enable_affinity_reasons:
            rules_block = (
                "- For each required ID, return an object with keys 'score' and 'reason'.\n"
                "- The reason must be short and specific to that ID.\n"
            )
            example_block = (
                '{"<ID_FROM_INPUT_1>": {"score": 85, "reason": "brief reason"}, "<ID_FROM_INPUT_2>": {"score": 42, "reason": "brief reason"}}\n\n'
            )
        else:
            rules_block = "- Values must be numbers in [0,120].\n"
            example_block = '{"<ID_FROM_INPUT_1>": 85, "<ID_FROM_INPUT_2>": 42}\n\n'
        return (
            "You are a power systems expert evaluator assessing research scope. "
            "Evaluate the affinity between the abstract and a set of Research Agenda (RA) questions. "
            "Score the abstract’s relevance to the research question from 0–120 using these ranges: 1–40 = Low relevance (0 = no relevance; topics may be tangentially related but do not directly address the questions’s goals); 41–80 = Moderate relevance (clear connection, but not central—e.g., focuses on methods or secondary aspects rather than the questions’s core problem); 81–120 = High relevance (directly and substantially addresses the questions’s main objectives).\n"
            "Return ONLY one JSON object (no extra text).\n"
            "Rules:\n"
            "- Include ALL required IDs exactly once as keys; do not add/rename keys.\n"
            f"{rules_block}"
            f"Required IDs: {required_ids_text}\n"
            f"{few_shot_block}"
            f"Abstract:\n{abstract}\n\n"
            f"Research questions:\n{targets_text}\n\n"
            "Output JSON example:\n"
            f"{example_block}"
        )

    def _build_prp_scope_prompt(
        self,
        abstract: str,
        targets_text: str,
        general_prp_description: str,
        required_ids: list[str],
    ) -> str:
        required_ids_text = ", ".join(required_ids)
        few_shot_block = self._format_few_shot_block("PRP")
        if self.enable_affinity_reasons:
            affinity_scores_example = (
                '    "<ID_FROM_INPUT_1>": {"score": 0, "reason": "brief reason"},\n'
                '    "<ID_FROM_INPUT_2>": {"score": 0, "reason": "brief reason"}\n'
            )
            affinity_scores_rule = "- affinity_scores values must be objects with keys 'score' and 'reason'.\n"
        else:
            affinity_scores_example = (
                '    "<ID_FROM_INPUT_1>": 0,\n'
                '    "<ID_FROM_INPUT_2>": 0\n'
            )
            affinity_scores_rule = "- affinity_scores values must be numbers in [0,100].\n"
        return (
            "You are a power systems expert evaluator of research scope. "
            "Evaluate the affinity between the abstract and a set of Primary Research Programmes (PRP). "
            "Score the abstract’s relevance to the program from 0–120 using these ranges: 1–40 = Low relevance (0 = no relevance; topics may be tangentially related but do not directly address the program’s goals); 41–80 = Moderate relevance (clear connection, but not central—e.g., focuses on methods or secondary aspects rather than the program’s core problem); 81–120 = High relevance (directly and substantially addresses the program’s main objectives).\n"
            "An abstract can be outside PRP scope.\n\n"
            "Return ONLY one JSON object (no extra text) with this schema:\n"
            "{\n"
            "  \"belongs_any_prp\": true,\n"
            "  \"membership_confidence\": 0,\n"
            "  \"affinity_scores\": {\n"
            f"{affinity_scores_example}"
            "  }\n"
            "}\n\n"
            "Rules:\n"
            "- belongs_any_prp: boolean.\n"
            "- membership_confidence: number in [0,100], and >0 if belongs_any_prp=true.\n"
            "- affinity_scores: include ALL required IDs exactly once; do not add or rename keys.\n"
            f"{affinity_scores_rule}"
            f"Required IDs: {required_ids_text}\n\n"
            f"{few_shot_block}"
            f"Abstract:\n{abstract}\n\n"
            f"General PRP description (scope anchor):\n{general_prp_description}\n\n"
            f"Primary Research Programmes:\n{targets_text}\n\n"
        )

    def _rate_affinity_batch_single_call(self, abstract: str, targets: list, target_type: str) -> dict:
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
            f"- [{t['id']}] {t['text']}"
            for t in normalized_targets
        )
        required_ids = [t["id"] for t in normalized_targets]
        prompt = self._build_affinity_prompt(abstract, targets_text, target_type, required_ids)
        if self.affinity_debug:
            print(f"[rate_affinity_batch] prompt:\n{prompt}")
        data = self._call_json_prompt_with_retries(
            prompt,
            temperature=0.0,
            debug=self.affinity_debug,
            log_tag="rate_affinity_batch",
        )
        if not isinstance(data, dict):
            return {}
        return self._normalize_batch_scores(data, targets, enable_reasons=self.enable_affinity_reasons)

    def rate_affinity(self, abstract: str, target_text: str, target_type: str = "RA"):
        if not abstract or not target_text:
            return None

        targets = [{"id": "single", "text": target_text}]
        result = self.rate_affinity_batch(abstract, targets, target_type)
        return result.get("single")

    def rate_affinity_batch(self, abstract: str, targets: list, target_type: str = "RA") -> dict:
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

        scores = self._normalize_batch_scores(
            affinity_scores_raw,
            targets,
            enable_reasons=self.enable_affinity_reasons,
        )
        expected_ids = [self._normalize_target_id(t.get("id")) for t in targets]
        id_to_text = {self._normalize_target_id(t.get("id")): str(t.get("text", "")) for t in targets}

        missing_ids = [tid for tid in expected_ids if tid not in scores]
        if missing_ids:
            missing_questions = [{"id": tid, "text": id_to_text.get(tid, "")} for tid in missing_ids]
            raise ValueError(f"Missing affinity scores for targets: {missing_questions}")

        invalid_ids = [
            tid
            for tid, val in scores.items()
            if val is None or (isinstance(val, dict) and val.get("score") is None)
        ]
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
            f"- [{t['id']}] {t['text']}"
            for t in normalized_targets
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

                message_content = getattr(resp.choices[0].message, "content", None) if getattr(resp, "choices", None) else None
                reply = str(message_content).strip() if message_content is not None else str(resp)

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