"""Select one or more (abstract, question) pairs per Ministral score sub-band
where a bimodal model is maximally divergent from Ministral's score.

Usage (as module):
  from select_ministral_divergent_pairs import select_divergent_pairs
  selected_df = select_divergent_pairs(df, ...)

Also provides a CLI for running against a CSV.
"""
from typing import List, Tuple, Dict, Optional
import argparse
import logging
import pandas as pd
import numpy as np
import os
import glob

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

DEFAULT_ANCHOR = "ministral-3-14b-cloud"
DEFAULT_BIMODAL_PRIORITY = [
    "gemini-3.1-flash-lite-preview",
    "gpt-5.4-mini",
    "gemma4-31b-cloud",
    "mistral-large-3-675b-cloud",
    "gpt-5.4",
]

# Explicit run id for main auto-run. Kept explicit here to avoid embedding it as
# a misleading argparse default (user must pass --run-id to override).
RUN_ID = "20260422_212121"


def _map_sub_band(score: float) -> Optional[str]:
    if pd.isna(score):
        return None
    s = float(score)
    if 0 <= s <= 20:
        return "0-20"
    if 21 <= s <= 40:
        return "21-40"
    if 41 <= s <= 55:
        return "41-55"
    if 56 <= s <= 70:
        return "56-70"
    if 71 <= s <= 85:
        return "71-85"
    if 86 <= s <= 100:
        return "86-100"
    return None


def select_divergent_pairs(
    df: pd.DataFrame,
    anchor_model: str = DEFAULT_ANCHOR,
    bimodal_priority: List[str] = DEFAULT_BIMODAL_PRIORITY,
    abstract_id_col: str = "abstract_id",
    question_id_col: str = "question_id",
    model_col: str = "model",
    score_col: str = "score",
    level_col: str = "level",
    abstract_text_col: Optional[str] = None,
    question_text_col: Optional[str] = None,
    output_csv: str = "ministral_anchored_examples.csv",
    buffer_default: int = 3,
    per_band: int = 1,
) -> Tuple[pd.DataFrame, Dict]:
    """Run the selection pipeline on `df` and write `output_csv`.

    Returns (selected_df, metadata)
    """
    if per_band < 1:
        raise ValueError("per_band must be at least 1")

    models = [anchor_model] + bimodal_priority

    relaxed_sub_bands: List[str] = []
    no_candidate_sub_bands: List[str] = []

    total_before = len(df)

    # Keep only rows for the 6 models
    df_models = df[df[model_col].isin(models)].copy()

    # Pivot scores and levels to wide
    score_wide = df_models.pivot_table(
        index=[abstract_id_col, question_id_col],
        columns=model_col,
        values=score_col,
        aggfunc="mean",
    )
    level_wide = df_models.pivot_table(
        index=[abstract_id_col, question_id_col],
        columns=model_col,
        values=level_col,
        aggfunc=lambda x: x.mode().iat[0] if len(x.mode()) > 0 else (x.iloc[0] if len(x) > 0 else np.nan),
    )

    # Rename columns
    score_wide = score_wide.rename(columns=lambda c: f"score_{c}")
    level_wide = level_wide.rename(columns=lambda c: f"level_{c}")

    wide = pd.concat([score_wide, level_wide], axis=1)

    # Ensure anchor column exists
    anchor_score_col = f"score_{anchor_model}"
    if anchor_score_col not in wide.columns:
        logger.warning("Anchor model %s not present in data.", anchor_model)

    # majority vote across all 6 models' level_{model}
    level_cols = [f"level_{m}" for m in models if f"level_{m}" in wide.columns]

    def majority_vote(row):
        vals = [v for v in row[level_cols].tolist() if pd.notna(v)]
        if not vals:
            return "disputed"
        counts = pd.Series(vals).value_counts()
        if len(counts) == 0:
            return "disputed"
        top = counts.iloc[0]
        winners = counts[counts == top].index.tolist()
        if len(winners) > 1:
            return "disputed"
        return winners[0]

    wide = wide.reset_index()
    wide["majority_vote_level"] = wide.apply(majority_vote, axis=1)

    total_after_vote = len(wide)

    # Filter: majority_vote_level == 'moderate' and not disputed and score_ministral between 0 and 100
    wide_filtered = wide[wide["majority_vote_level"] == "moderate"].copy()
    # Ensure anchor score column exists in filtered
    if anchor_score_col in wide_filtered.columns:
        # pandas changed the `inclusive` parameter to accept strings in newer versions
        try:
            wide_filtered = wide_filtered[wide_filtered[anchor_score_col].between(0, 100, inclusive='both')]
        except TypeError:
            wide_filtered = wide_filtered[wide_filtered[anchor_score_col].between(0, 100)]
    else:
        # Can't filter by anchor score if missing; keep rows but log
        logger.warning("Anchor score column %s missing; skipping anchor score bounds filter.", anchor_score_col)

    total_after_majority = len(wide_filtered)

    # Assign sub-bands
    wide_filtered["score_ministral"] = wide_filtered.get(anchor_score_col)
    wide_filtered["sub_band"] = wide_filtered["score_ministral"].map(_map_sub_band)

    # Exclude rows within buffer of boundaries
    boundaries = [20, 40, 55, 70, 85]
    buffer = buffer_default

    def exclude_buffer(df_in, buffers_map: Dict[str, int]):
        df = df_in.copy()
        keep_mask = []
        for _, row in df.iterrows():
            s = row["score_ministral"]
            if pd.isna(s):
                keep_mask.append(False)
                continue
            close = False
            for b in boundaries:
                if abs(s - b) <= buffers_map.get(row["sub_band"], buffer):
                    close = True
                    break
            keep_mask.append(not close)
        return df[pd.Series(keep_mask, index=df.index)]

    buffers_map = {}
    # initial exclusion
    wide_buffered = exclude_buffer(wide_filtered, buffers_map)

    # Check sub-band counts and relax if needed
    sub_bands = ["0-20", "21-40", "41-55", "56-70", "71-85", "86-100"]
    counts = wide_buffered["sub_band"].value_counts().to_dict()
    relaxed = []
    for sb in sub_bands:
        if counts.get(sb, 0) == 0:
            # relax this sub-band: set buffer 2 for this band
            buffers_map[sb] = 2
            relaxed.append(sb)

    if relaxed:
        relaxed_sub_bands = relaxed
        wide_buffered = exclude_buffer(wide_filtered, buffers_map)

    total_after_subband = len(wide_filtered)
    total_after_buffer = len(wide_buffered)

    # Compute deviations for bimodal models
    for m in bimodal_priority:
        score_m = f"score_{m}"
        dev = f"deviation_{m}"
        abs_dev = f"abs_deviation_{m}"
        if score_m in wide_buffered.columns:
            wide_buffered[dev] = wide_buffered[score_m] - wide_buffered["score_ministral"]
            wide_buffered[abs_dev] = wide_buffered[dev].abs()
        else:
            wide_buffered[dev] = np.nan
            wide_buffered[abs_dev] = np.nan

    # Selection per sub-band
    selections = []
    for sb in sub_bands:
        sub_df = wide_buffered[wide_buffered["sub_band"] == sb]
        if sub_df.empty:
            no_candidate_sub_bands.append(sb)
            continue
        band_selected = []
        seen_pairs = set()
        model_candidates = {}
        for i, m in enumerate(bimodal_priority, start=1):
            abs_col = f"abs_deviation_{m}"
            if abs_col not in sub_df.columns:
                continue

            sub_df_valid = sub_df[sub_df[abs_col].notna()]
            if sub_df_valid.empty:
                continue

            model_candidates[m] = {
                "priority": i,
                "abs_col": abs_col,
                "rows": sub_df_valid.sort_values(
                    by=[abs_col, abstract_id_col, question_id_col],
                    ascending=[False, True, True],
                ).to_dict("records"),
                "cursor": 0,
            }

        if not model_candidates:
            no_candidate_sub_bands.append(sb)
            continue

        used_models = set()
        while len(band_selected) < per_band:
            if len(used_models) < len(model_candidates):
                pass_models = [m for m in bimodal_priority if m in model_candidates and m not in used_models]
            else:
                pass_models = [m for m in bimodal_priority if m in model_candidates]

            selected_this_round = False
            for m in pass_models:
                model_state = model_candidates[m]
                rows = model_state["rows"]
                cursor = model_state["cursor"]
                abs_col = model_state["abs_col"]
                priority = model_state["priority"]

                while cursor < len(rows):
                    cand = rows[cursor]
                    cursor += 1
                    pair_key = (cand[abstract_id_col], cand[question_id_col])
                    if pair_key in seen_pairs:
                        continue
                    rec = dict(cand)
                    rec.update({
                        "selection_rank": len(band_selected) + 1,
                        "selection_priority": priority,
                        "primary_divergent_model": m,
                        "max_abs_deviation": float(cand[abs_col]),
                    })
                    band_selected.append(rec)
                    seen_pairs.add(pair_key)
                    used_models.add(m)
                    selected_this_round = True
                    break

                model_state["cursor"] = cursor
                if len(band_selected) >= per_band:
                    break

            if not selected_this_round:
                break

        if band_selected:
            selections.extend(band_selected)
        else:
            no_candidate_sub_bands.append(sb)

    if not selections:
        logger.info("No selections found.")
        selected_df = pd.DataFrame(columns=["sub_band", abstract_id_col, question_id_col])
    else:
        selected_df = pd.DataFrame(selections)

    # Retrieve text columns by merging back to original df: use first occurrence per id pair
    id_pairs = df[[abstract_id_col, question_id_col]].drop_duplicates()
    # attempt to find text columns
    if abstract_text_col is None or question_text_col is None:
        # heuristics: find any column containing 'abstract' and not id, and 'question' and not id
        cols = list(df.columns)
        if abstract_text_col is None:
            candidates = [c for c in cols if "abstract" in c.lower() and c not in [abstract_id_col]]
            abstract_text_col = candidates[0] if candidates else None
        if question_text_col is None:
            candidates = [c for c in cols if "question" in c.lower() and c not in [question_id_col]]
            question_text_col = candidates[0] if candidates else None

    if abstract_text_col and question_text_col:
        text_df = df[[abstract_id_col, question_id_col, abstract_text_col, question_text_col]].drop_duplicates(subset=[abstract_id_col, question_id_col])
        selected_df = selected_df.merge(text_df, on=[abstract_id_col, question_id_col], how="left")
        selected_df = selected_df.rename(columns={
            abstract_text_col: "abstract_text",
            question_text_col: "question_text",
        })
    else:
        selected_df["abstract_text"] = np.nan
        selected_df["question_text"] = np.nan

    # Prepare export columns
    export_cols = [
        "sub_band",
        abstract_id_col,
        question_id_col,
        "abstract_text",
        "question_text",
        "score_ministral",
        "selection_rank",
        "selection_priority",
        "primary_divergent_model",
        "max_abs_deviation",
        "majority_vote_level",
    ]

    # add score_{model} for each of 6 models
    for m in models:
        c = f"score_{m}"
        if c in wide_buffered.columns:
            export_cols.append(c)
        else:
            export_cols.append(c)
            selected_df[c] = np.nan

    # add deviation_{model} for each of 5 other models (bimodal_priority list)
    for m in bimodal_priority:
        export_cols.append(f"deviation_{m}")

    # Ensure these columns exist in selected_df (they came from wide index)
    # Many fields are in selected_df already since selections were taken from wide_buffered

    # Reorder and fill missing columns
    for c in export_cols:
        if c not in selected_df.columns:
            selected_df[c] = np.nan

    selected_df = selected_df[export_cols]

    # Sort by sub_band according to defined order
    band_order = {b: i for i, b in enumerate(sub_bands)}
    selected_df["_band_order"] = selected_df["sub_band"].map(band_order)
    sort_cols = ["_band_order"]
    if "selection_rank" in selected_df.columns:
        sort_cols.append("selection_rank")
    selected_df = selected_df.sort_values(sort_cols).drop(columns=["_band_order"]) 

    # write CSV
    selected_df.to_csv(output_csv, index=False)
    logger.info("Wrote %d selected rows to %s", len(selected_df), output_csv)

    meta: Dict[str, object] = {
        "relaxed_sub_bands": relaxed_sub_bands,
        "no_candidate_sub_bands": no_candidate_sub_bands,
        "total_before": total_before,
        "total_after_vote": total_after_vote,
        "total_after_majority": total_after_majority,
        "total_after_subband": total_after_subband,
        "total_after_buffer": total_after_buffer,
        "final_selected": len(selected_df),
    }

    return selected_df, meta


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("input_csv", nargs="?", default=None, help="input CSV with columns for model, score, level, ids. If omitted, runs standard benchmark lookup for the timestamp in --run-id")
    p.add_argument("--output", default=None, help="output CSV path (if omitted, defaults to data/results/affinity_benchmark/{run_id}/ministral_anchored_examples.csv when using auto run)")
    p.add_argument("--anchor", default=DEFAULT_ANCHOR)
    p.add_argument("--abstract-id", default="abstract_id")
    p.add_argument("--question-id", default="question_id")
    p.add_argument("--model-col", default="model")
    p.add_argument("--score-col", default="score")
    p.add_argument("--level-col", default="level")
    p.add_argument("--run-id", dest="run_id", default=None, help="benchmark run timestamp to use when input_csv is omitted (no default shown; script has an explicit RUN_ID)")
    p.add_argument("--save-dir", default="data/processed", help="directory to save output when auto-running")
    p.add_argument("--per-band", type=int, default=1, help="number of pairs to select per Ministral score sub-band")
    args = p.parse_args()
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        out_path = args.output or "ministral_anchored_examples.csv"
        selected, meta = select_divergent_pairs(
            df,
            anchor_model=args.anchor,
            bimodal_priority=DEFAULT_BIMODAL_PRIORITY,
            abstract_id_col=args.abstract_id,
            question_id_col=args.question_id,
            model_col=args.model_col,
            score_col=args.score_col,
            level_col=args.level_col,
            output_csv=out_path,
            per_band=args.per_band,
        )
        print(meta)
        return

    # Auto-run mode: gather per-model benchmark CSVs from affinity_benchmark/{run}/{model}/ra_affinities.csv
    # Use explicit RUN_ID when --run-id is omitted (argparse shows no default)
    run_ts = args.run_id if args.run_id is not None else RUN_ID
    models = [args.anchor] + DEFAULT_BIMODAL_PRIORITY
    files_found = []
    for m in models:
        candidate = f"data/results/affinity_benchmark/{run_ts}/{m}/ra_affinities.csv"
        if os.path.exists(candidate):
            files_found.append((m, candidate))
            continue
        # fallback: glob search for file containing run_ts and model name
        pats = glob.glob(f"data/results/**/{m}/*ra_affinities*.csv", recursive=True)
        pats = [p for p in pats if run_ts in p]
        if pats:
            files_found.append((m, sorted(pats)[-1]))
        else:
            logger.warning("No affinity file found for model %s and run %s", m, run_ts)

    if not files_found:
        logger.error("No benchmark files found for run %s", run_ts)
        return

    rows = []
    for m, p in files_found:
        try:
            d = pd.read_csv(p)
        except Exception as e:
            logger.warning("Failed to read %s: %s", p, e)
            continue
        d = d.copy()
        d['model'] = m
        # normalize score column to LLM_Affinity if present
        if 'LLM_Affinity' in d.columns:
            d['score_for_script'] = d['LLM_Affinity']
        else:
            # try alternatives
            for alt in ['LLM_AFFINITY', 'llm_affinity', 'Affinity', 'score']:
                if alt in d.columns:
                    d['score_for_script'] = d[alt]
                    break
        # compute level using heuristic
        def level_of(s):
            try:
                s = float(s)
            except Exception:
                return ''
            if s <= 40:
                return 'low'
            if s < 70:
                return 'moderate'
            return 'high'
        d['level_for_script'] = d['score_for_script'].apply(level_of)
        # ensure id cols exist
        keep_cols = []
        for c in [args.abstract_id, args.question_id, 'RA_Question', 'RA2025_ID', 'Abstract_Index']:
            if c in d.columns and c not in keep_cols:
                keep_cols.append(c)
        # prefer canonical names
        # assemble minimal frame with expected names
        frame = pd.DataFrame()
        # map abstract id
        if args.abstract_id in d.columns:
            frame[args.abstract_id] = d[args.abstract_id]
        elif 'Abstract_Index' in d.columns:
            frame[args.abstract_id] = d['Abstract_Index']
        # map question id
        if args.question_id in d.columns:
            frame[args.question_id] = d[args.question_id]
        elif 'RA2025_ID' in d.columns:
            frame[args.question_id] = d['RA2025_ID']
        # RA_Question text
        if 'RA_Question' in d.columns:
            frame['RA_Question'] = d['RA_Question']
        # model, score, level
        frame['model'] = d['model']
        frame['score'] = d.get('score_for_script')
        frame['level'] = d.get('level_for_script')
        rows.append(frame)

    if not rows:
        logger.error("No data frames assembled for run %s", run_ts)
        return

    combined = pd.concat(rows, ignore_index=True)
    out_name = args.output or "ministral_anchored_examples.csv"
    out_dir = os.path.join("data", "results", "affinity_benchmark", run_ts)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name) if args.output is None else args.output

    selected, meta = select_divergent_pairs(
        combined,
        anchor_model=args.anchor,
        bimodal_priority=DEFAULT_BIMODAL_PRIORITY,
        abstract_id_col=args.abstract_id,
        question_id_col=args.question_id,
        model_col='model',
        score_col='score',
        level_col='level',
        output_csv=out_path,
        per_band=args.per_band,
    )
    print(meta)
    print('Wrote:', out_path)


if __name__ == "__main__":
    _cli()
