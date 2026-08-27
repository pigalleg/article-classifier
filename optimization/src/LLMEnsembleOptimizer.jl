module LLMEnsembleOptimizer

using CSV
using DataFrames
using Gurobi
using JuMP
using Statistics
import MathOptInterface as MOI

const PAIR_KEYS = [:Abstract_Index, :RA2025_ID]
const EXPERT_COLUMNS = [:Abstract_Index, :RA2025_ID, :Evaluator_Affinity]
const BENCHMARK_COLUMNS = [:Abstract_Index, :RA2025_ID, :Model_Slug, :LLM_Affinity]
const AFFINITY_LEVELS = ["Low", "Moderate", "High"]

struct OptimizerConfig
    objective::Symbol
    low_multiplier::Float64
    moderate_multiplier::Float64
    high_multiplier::Float64
end

OptimizerConfig(; objective::Symbol = :weighted_mae, low_multiplier::Real = 1.0,
                moderate_multiplier::Real = 2.0, high_multiplier::Real = 2.0) =
    OptimizerConfig(objective, Float64(low_multiplier), Float64(moderate_multiplier), Float64(high_multiplier))

function require_columns(data::DataFrame, columns::Vector{Symbol}, source::AbstractString)
    missing_columns = setdiff(columns, Symbol.(names(data)))
    isempty(missing_columns) || error("$source is missing required columns: $(join(string.(missing_columns), ", "))")
end

function load_dataset(expert_path::AbstractString, benchmark_path::AbstractString)
    experts = CSV.read(expert_path, DataFrame)
    benchmark = CSV.read(benchmark_path, DataFrame)
    require_columns(experts, EXPERT_COLUMNS, "Expert file")
    require_columns(benchmark, BENCHMARK_COLUMNS, "Benchmark file")

    available_models = unique(benchmark.Model_Slug[.!ismissing.(benchmark.LLM_Affinity)])
    excluded_models = setdiff(unique(benchmark.Model_Slug), available_models)
    benchmark = filter(:Model_Slug => in(available_models), benchmark)
    isempty(available_models) && error("Benchmark file contains no non-missing model scores.")

    duplicate_scores = combine(groupby(benchmark, vcat(PAIR_KEYS, :Model_Slug)), nrow => :count)
    any(duplicate_scores.count .> 1) && error("Benchmark file has duplicate pair/model scores.")
    duplicate_experts = combine(groupby(experts, PAIR_KEYS), nrow => :count)
    any(duplicate_experts.count .> 1) && error("Expert file has duplicate abstract/question pairs.")

    wide_scores = unstack(benchmark[:, BENCHMARK_COLUMNS], PAIR_KEYS, :Model_Slug, :LLM_Affinity)
    model_columns = setdiff(Symbol.(names(wide_scores)), PAIR_KEYS)
    isempty(model_columns) && error("Benchmark file contains no model scores.")
    joined = innerjoin(experts[:, EXPERT_COLUMNS], wide_scores, on = PAIR_KEYS)
    complete_rows = dropmissing(joined, vcat([:Evaluator_Affinity], model_columns))
    nrow(complete_rows) > 0 || error("No complete expert/model-score rows remain after joining inputs.")

    target = Float64.(complete_rows.Evaluator_Affinity)
    scores = Matrix{Float64}(complete_rows[:, model_columns])
    return (data = complete_rows, target = target, scores = scores, model_names = String.(model_columns),
            excluded_models = String.(excluded_models), joined_rows = nrow(joined), retained_rows = nrow(complete_rows))
end

function solve_weights(scores::Matrix{Float64}, target::Vector{Float64}, config::OptimizerConfig = OptimizerConfig())
    config.objective == :weighted_mae || error("Unsupported objective: $(config.objective)")
    size(scores, 1) == length(target) || error("Score rows and target values must have equal length.")
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    pair_count, model_count = size(scores)
    @variable(model, weights[1:model_count] >= 0)
    @constraint(model, sum(weights) == 1)
    @expression(model, ensemble_scores[i = 1:pair_count], sum(weights[m] * scores[i, m] for m = 1:model_count))
    @variable(model, errors[1:pair_count] >= 0)
    @constraint(model, [i = 1:pair_count], errors[i] >= ensemble_scores[i] - target[i])
    @constraint(model, [i = 1:pair_count], errors[i] >= target[i] - ensemble_scores[i])
    level_weights = sample_weights(target, config)
    @objective(model, Min, sum(level_weights[i] * errors[i] for i = 1:pair_count))
    optimize!(model)
    status = termination_status(model)
    status == MOI.OPTIMAL || error("Optimization failed with status $status.")
    fitted_scores = scores * value.(weights)
    return (weights = value.(weights), fitted_scores = fitted_scores, objective_value = objective_value(model), status = string(status))
end

function grouped_folds(groups::AbstractVector, fold_count::Integer)
    fold_count >= 2 || error("fold_count must be at least 2.")
    unique_groups = sort(unique(string.(groups)))
    length(unique_groups) >= fold_count || error("fold_count exceeds the number of unique abstracts.")
    fold_by_group = Dict(group => mod(index - 1, fold_count) + 1 for (index, group) in enumerate(unique_groups))
    return [fold_by_group[string(group)] for group in groups]
end

function affinity_level(score::Real)
    score <= 40 && return "Low"
    score <= 80 && return "Moderate"
    return "High"
end

function sample_weights(target::AbstractVector, config::OptimizerConfig)
    all((config.low_multiplier, config.moderate_multiplier, config.high_multiplier) .> 0) ||
        error("All level multipliers must be positive.")
    return [score <= 40 ? config.low_multiplier : score <= 80 ? config.moderate_multiplier : config.high_multiplier
            for score in target]
end

function classification_metrics(expert_levels::AbstractVector, predicted_levels::AbstractVector)
    pair_count = length(expert_levels)
    pair_count == length(predicted_levels) || error("Expert and predicted levels must have equal length.")
    confusion = [count((expert_levels .== actual) .& (predicted_levels .== predicted))
                 for actual in AFFINITY_LEVELS, predicted in AFFINITY_LEVELS]
    expected = sum(confusion, dims = 2) * sum(confusion, dims = 1) / pair_count
    quadratic_weights = [(actual - predicted)^2 / (length(AFFINITY_LEVELS) - 1)^2
                         for actual in eachindex(AFFINITY_LEVELS), predicted in eachindex(AFFINITY_LEVELS)]
    expected_disagreement = sum(quadratic_weights .* expected)
    quadratic_weighted_kappa = expected_disagreement == 0 ? NaN :
        1 - sum(quadratic_weights .* confusion) / expected_disagreement
    rows = NamedTuple[(Metric = "Quadratic_Weighted_Kappa", Value = quadratic_weighted_kappa)]
    for (index, level) in enumerate(AFFINITY_LEVELS)
        true_positive = confusion[index, index]
        false_positive = sum(confusion[:, index]) - true_positive
        false_negative = sum(confusion[index, :]) - true_positive
        true_negative = pair_count - true_positive - false_positive - false_negative
        precision = true_positive + false_positive == 0 ? 0.0 : true_positive / (true_positive + false_positive)
        recall = true_positive + false_negative == 0 ? 0.0 : true_positive / (true_positive + false_negative)
        one_vs_rest_accuracy = (true_positive + true_negative) / pair_count
        append!(rows, [
            (Metric = "$(level)_Support", Value = Float64(true_positive + false_negative)),
            (Metric = "$(level)_Precision", Value = precision),
            (Metric = "$(level)_Recall", Value = recall),
            (Metric = "$(level)_One_vs_Rest_Accuracy", Value = one_vs_rest_accuracy),
            (Metric = "$(level)_True_Positive", Value = Float64(true_positive)),
            (Metric = "$(level)_False_Positive", Value = Float64(false_positive)),
            (Metric = "$(level)_False_Negative", Value = Float64(false_negative)),
            (Metric = "$(level)_True_Negative", Value = Float64(true_negative)),
        ])
    end
    return rows
end

function cross_validate(dataset; fold_count::Integer = 5, config::OptimizerConfig = OptimizerConfig())
    folds = grouped_folds(dataset.data.Abstract_Index, fold_count)
    prediction_frames = DataFrame[]
    weight_frames = DataFrame[]
    for fold in 1:fold_count
        train_indices = findall(!=(fold), folds)
        test_indices = findall(==(fold), folds)
        result = solve_weights(dataset.scores[train_indices, :], dataset.target[train_indices], config)
        test_scores = dataset.scores[test_indices, :] * result.weights
        predictions = hcat(dataset.data[test_indices, PAIR_KEYS], dataset.data[test_indices, Symbol.(dataset.model_names)])
        predictions.Fold = fill(fold, length(test_indices))
        predictions.Evaluator_Affinity = dataset.target[test_indices]
        predictions.Weighted_Affinity = test_scores
        predictions.Equal_Weight_Affinity = vec(mean(dataset.scores[test_indices, :], dims = 2))
        predictions.Expert_Level = affinity_level.(predictions.Evaluator_Affinity)
        predictions.Predicted_Level = affinity_level.(predictions.Weighted_Affinity)
        push!(prediction_frames, predictions)
        push!(weight_frames, DataFrame(Fold = fill(fold, length(result.weights)), Model_Slug = dataset.model_names,
            Weight = result.weights, Train_Pairs = fill(length(train_indices), length(result.weights)),
            Test_Pairs = fill(length(test_indices), length(result.weights)), Objective_Value = fill(result.objective_value, length(result.weights)),
            Solver_Status = fill(result.status, length(result.weights))))
    end
    predictions = vcat(prediction_frames...)
    return (predictions = predictions, weights = vcat(weight_frames...), metrics = metrics_summary(predictions, config))
end

function metrics_summary(predictions::DataFrame, config::OptimizerConfig)
    expert_scores = Float64.(predictions.Evaluator_Affinity)
    level_weights = sample_weights(expert_scores, config)
    approaches = [(:Optimized, Float64.(predictions.Weighted_Affinity)), (:Equal_Weight, Float64.(predictions.Equal_Weight_Affinity))]
    rows = NamedTuple[]
    for (name, scores) in approaches
        errors = scores .- expert_scores
        push!(rows, (Approach = String(name), Metric = "Weighted_MAE", Value = sum(level_weights .* abs.(errors)) / sum(level_weights)))
        push!(rows, (Approach = String(name), Metric = "MAE", Value = sum(abs.(errors)) / length(errors)))
        push!(rows, (Approach = String(name), Metric = "Mean_Signed_Error", Value = sum(errors) / length(errors)))
        predicted_levels = affinity_level.(scores)
        push!(rows, (Approach = String(name), Metric = "Three_Level_Accuracy", Value = sum(predicted_levels .== predictions.Expert_Level) / length(scores)))
        append!(rows, [(Approach = String(name), metric.Metric, metric.Value)
                       for metric in classification_metrics(predictions.Expert_Level, predicted_levels)])
    end
    return DataFrame(rows)
end

function write_results(results, output_dir::AbstractString)
    mkpath(output_dir)
    CSV.write(joinpath(output_dir, "fold_weights.csv"), results.weights)
    CSV.write(joinpath(output_dir, "out_of_fold_predictions.csv"), results.predictions)
    CSV.write(joinpath(output_dir, "metrics_summary.csv"), results.metrics)
end

export OptimizerConfig, affinity_level, cross_validate, grouped_folds, load_dataset, metrics_summary, sample_weights, solve_weights, write_results

end
