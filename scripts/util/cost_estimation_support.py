from __future__ import annotations

import numpy as np
import pandas as pd

COST_EXTRAPOLATION_ABSTRACT_COUNT = 100

# Standard synchronous API pricing in USD per 1 million tokens.
# Ollama remains unknown because no runtime, hardware, or electricity data
# were available in the source document.
MODEL_PRICING_USD: dict[str, dict[str, float | str]] = {
    # xAI pricing tier used here: grok-4.20-0309-reasoning with prompt length
    # < 200k tokens.
    # Source: https://docs.x.ai/developers/pricing
    "grok-4.20-0309-reasoning": {
        "pricing_mode": "standard_api",
        "input_usd_per_million_tokens": 1.25,
        "output_usd_per_million_tokens": 2.50,
    },
    "gemini-3.1-flash-lite": {
        "pricing_mode": "standard_api",
        "input_usd_per_million_tokens": 0.25,
        "output_usd_per_million_tokens": 1.50,
    },
    "gpt-5.4-mini": {
        "pricing_mode": "standard_api",
        "input_usd_per_million_tokens": 0.75,
        "output_usd_per_million_tokens": 4.50,
    },
    "gpt-5.4": {
        "pricing_mode": "standard_api",
        "input_usd_per_million_tokens": 2.50,
        "output_usd_per_million_tokens": 15.00,
    },
}


def calculate_api_cost_usd(
    model: str,
    input_tokens: float,
    output_tokens: float,
) -> float:
    """Calculate standard API cost in USD from token usage."""
    if (
        model not in MODEL_PRICING_USD
        or pd.isna(input_tokens)
        or pd.isna(output_tokens)
    ):
        return np.nan

    rates = MODEL_PRICING_USD[model]

    input_cost_usd = (
        input_tokens
        / 1_000_000
        * float(rates["input_usd_per_million_tokens"])
    )
    output_cost_usd = (
        output_tokens
        / 1_000_000
        * float(rates["output_usd_per_million_tokens"])
    )

    return input_cost_usd + output_cost_usd


def build_cost_estimation_df() -> pd.DataFrame:
    """Build the benchmark/adjudication cost table and derived metrics."""
    cost_estimation_df = pd.DataFrame(
        [
            {
                "stage": "Benchmark",
                "model": "gemini-3.1-flash-lite",
                "pricing_mode": "standard_api",
                "currency": "USD",
                "abstract_count": 100,
                "examples_prompt": 4,
                "few_shot_max_examples_per_prompt": 4,
                "input_tokens": 602_000,
                "output_tokens": 152_000,
            },
            {
                "stage": "Benchmark",
                "model": "gemini-3.1-flash-lite",
                "pricing_mode": "standard_api",
                "currency": "USD",
                "abstract_count": 100,
                "examples_prompt": 20,
                "few_shot_max_examples_per_prompt": 30,
                "input_tokens": 1_538_000,
                "output_tokens": 156_000,
            },
            {
                "stage": "Benchmark",
                "model": "gpt-5.4-mini",
                "pricing_mode": "standard_api",
                "currency": "USD",
                "abstract_count": 100,
                "examples_prompt": 4,
                "few_shot_max_examples_per_prompt": 4,
                "input_tokens": 497_181,
                "output_tokens": 135_412,
            },
            {
                "stage": "Benchmark",
                "model": "gpt-5.4-mini",
                "pricing_mode": "standard_api",
                "currency": "USD",
                "abstract_count": 100,
                "examples_prompt": 20,
                "few_shot_max_examples_per_prompt": 30,
                "input_tokens": 1_150_819,
                "output_tokens": 124_068,
            },
            # {
            #     "stage": "Adjudication",
            #     "model": "gpt-5.4",
            #     "pricing_mode": "standard_api",
            #     "currency": "USD",
            #     "abstract_count": 100,
            #     "examples_prompt": 4,
            #     "few_shot_max_examples_per_prompt": 4,
            #     "input_tokens": 621_261,
            #     "output_tokens": 127_415,
            # },
            # {
            #     "stage": "Adjudication",
            #     "model": "gpt-5.4",
            #     "pricing_mode": "standard_api",
            #     "currency": "USD",
            #     "abstract_count": 100,
            #     "examples_prompt": 20,
            #     "few_shot_max_examples_per_prompt": 30,
            #     "input_tokens": 2_065_739,
            #     "output_tokens": 146_175,
            # },
            {
                "stage": "Adjudication",
                "model": "gpt-5.4-mini",
                "pricing_mode": "standard_api",
                "currency": "USD",
                "abstract_count": 100,
                "examples_prompt": 4,
                "few_shot_max_examples_per_prompt": 4,
                "input_tokens": 2_208_000,
                "output_tokens": 227_198,
            },
            {
                "stage": "Adjudication",
                "model": "gpt-5.4-mini",
                "pricing_mode": "standard_api",
                "currency": "USD",
                "abstract_count": 100,
                "examples_prompt": 20,
                "few_shot_max_examples_per_prompt": 30,
                "input_tokens": 1_449_000,
                "output_tokens": 138_960,
            },
            {
                "stage": "Benchmark",
                "model": "ollama",
                "pricing_mode": "local_runtime_unknown",
                "currency": "USD",
                "abstract_count": 100,
                "examples_prompt": 4,
                "few_shot_max_examples_per_prompt": 4,
                "input_tokens": np.nan,
                "output_tokens": np.nan,
            },
            {
                "stage": "Benchmark",
                "model": "ollama",
                "pricing_mode": "local_runtime_unknown",
                "currency": "USD",
                "abstract_count": 100,
                "examples_prompt": 20,
                "few_shot_max_examples_per_prompt": 30,
                "input_tokens": np.nan,
                "output_tokens": np.nan,
            },
        ]
    )

    cost_estimation_df["estimated_cost_usd"] = cost_estimation_df.apply(
        lambda row: calculate_api_cost_usd(
            model=row["model"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
        ),
        axis=1,
    )

    cost_estimation_df["estimated_cost_per_abstract_usd"] = (
        cost_estimation_df["estimated_cost_usd"]
        / cost_estimation_df["abstract_count"]
    )

    # Backward-compatible columns used in existing notebook cells.
    cost_estimation_df["cost_per_100_abstracts"] = cost_estimation_df[
        "estimated_cost_usd"
    ]
    cost_estimation_df["cost_per_abstract"] = cost_estimation_df[
        "estimated_cost_per_abstract_usd"
    ]

    cost_estimation_df["estimated_cost_usd_display"] = (
        cost_estimation_df["estimated_cost_usd"].round(4)
    )
    cost_estimation_df["estimated_cost_per_abstract_usd_display"] = (
        cost_estimation_df["estimated_cost_per_abstract_usd"].round(5)
    )

    return cost_estimation_df
