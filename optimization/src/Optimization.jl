const PAIR_KEYS = [:Abstract_Index, :RA2025_ID]
const EXPERT_COLUMNS = [:Abstract_Index, :RA2025_ID, :Evaluator_Affinity]
const BENCHMARK_COLUMNS = [:Abstract_Index, :RA2025_ID, :Model_Slug, :LLM_Affinity]
const AFFINITY_LEVELS = ["Low", "Moderate", "High"]

struct OptimizerConfig
    objective::Symbol
    weight_scope::Symbol
    calibration_scope::Symbol
    low_multiplier::Float64
    moderate_multiplier::Float64
    high_multiplier::Float64
    solver::Symbol
    learning_rate::Float64
    max_iterations::Int
    patience::Int
    tolerance::Float64
    level_temperature::Float64
    restart_count::Int
end

OptimizerConfig(; objective::Symbol = :weighted_mae, weight_scope::Symbol = :global, calibration_scope::Symbol = :none,
                low_multiplier::Real = 1.0, moderate_multiplier::Real = 2.0, high_multiplier::Real = 2.0,
                solver::Symbol = :exact, learning_rate::Real = 0.05, max_iterations::Integer = 5000,
                patience::Integer = 200, tolerance::Real = 1e-10, level_temperature::Real = 5.0,
                restart_count::Integer = 1) =
    OptimizerConfig(objective, weight_scope, calibration_scope, Float64(low_multiplier), Float64(moderate_multiplier),
                    Float64(high_multiplier), solver, Float64(learning_rate), Int(max_iterations), Int(patience),
                    Float64(tolerance), Float64(level_temperature), Int(restart_count))

function validate_solver_objective(config::OptimizerConfig)
    config.solver in (:exact, :gradient) || error("Unsupported solver: $(config.solver)")
    if config.solver == :exact
        config.objective in (:weighted_mae, :weighted_mse) ||
            error("The exact solver supports weighted_mae and weighted_mse; $(config.objective) requires solver = :gradient.")
    else
        config.objective in (:weighted_mae, :weighted_mse, :soft_qwk) || error("Unsupported objective: $(config.objective)")
    end
    return nothing
end

function require_columns(data::DataFrame, columns::Vector{Symbol}, source::AbstractString)
    missing_columns = setdiff(columns, Symbol.(names(data)))
    isempty(missing_columns) || error("$source is missing required columns: $(join(string.(missing_columns), ", "))")
end

function load_dataset(expert_path::AbstractString, benchmark_path::AbstractString; programme_mapping_path::Union{Nothing, AbstractString} = nothing)
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
    if !isnothing(programme_mapping_path)
        programme_mapping = CSV.read(programme_mapping_path, DataFrame)
        require_columns(programme_mapping, [:RA2025, :Grouping], "Programme mapping file")
        programme_mapping = unique(programme_mapping[:, [:RA2025, :Grouping]])
        rename!(programme_mapping, :RA2025 => :RA2025_ID)
        joined = innerjoin(joined, programme_mapping, on = :RA2025_ID)
    end
    complete_rows = dropmissing(joined, vcat([:Evaluator_Affinity], model_columns))
    nrow(complete_rows) > 0 || error("No complete expert/model-score rows remain after joining inputs.")
    target = Float64.(complete_rows.Evaluator_Affinity)
    scores = Matrix{Float64}(complete_rows[:, model_columns])
    return (data = complete_rows, target = target, scores = scores, model_names = String.(model_columns),
            excluded_models = String.(excluded_models), joined_rows = nrow(joined), retained_rows = nrow(complete_rows))
end

function solve_weights(scores::Matrix{Float64}, target::Vector{Float64}, config::OptimizerConfig = OptimizerConfig())
    validate_solver_objective(config)
    config.weight_scope == :global || error("Global solver requires weight_scope = :global.")
    size(scores, 1) == length(target) || error("Score rows and target values must have equal length.")
    if config.solver == :gradient
        result = descend_weights(scores, target, fill(1, length(target)), 1, config)
        weights = vec(result.weights)
        return (weights = weights, fitted_scores = scores * weights, objective_value = result.objective_value,
                status = result.status, iterations = result.iterations)
    end
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    pair_count, model_count = size(scores)
    @variable(model, weights[1:model_count] >= 0)
    @constraint(model, sum(weights) == 1)
    @expression(model, ensemble_scores[i = 1:pair_count], sum(weights[m] * scores[i, m] for m = 1:model_count))
    level_weights = sample_weights(target, config)
    if config.objective == :weighted_mae
        @variable(model, absolute_errors[1:pair_count] >= 0)
        @constraint(model, [i = 1:pair_count], absolute_errors[i] >= ensemble_scores[i] - target[i])
        @constraint(model, [i = 1:pair_count], absolute_errors[i] >= target[i] - ensemble_scores[i])
        @objective(model, Min, sum(level_weights[i] * absolute_errors[i] for i = 1:pair_count))
    else
        @objective(model, Min, sum(level_weights[i] * (ensemble_scores[i] - target[i])^2 for i = 1:pair_count))
    end
    optimize!(model)
    status = termination_status(model)
    status == MOI.OPTIMAL || error("Optimization failed with status $status.")
    fitted_scores = scores * value.(weights)
    return (weights = value.(weights), fitted_scores = fitted_scores, objective_value = objective_value(model), status = string(status))
end

function solve_weights(scores::Matrix{Float64}, target::Vector{Float64}, programmes::AbstractVector, config::OptimizerConfig)
    validate_solver_objective(config)
    config.weight_scope == :programme || error("Programme solver requires weight_scope = :programme.")
    size(scores, 1) == length(target) || error("Score rows and target values must have equal length.")
    length(programmes) == length(target) || error("Programme values and target values must have equal length.")
    programme_names = sort(unique(string.(programmes)))
    programme_indices = index_programmes(programmes, programme_names)
    if config.solver == :gradient
        result = descend_weights(scores, target, programme_indices, length(programme_names), config)
        return (weights = result.weights, programme_names = programme_names,
                fitted_scores = programme_ensemble_scores(scores, programme_indices, result.weights),
                objective_value = result.objective_value, status = result.status, iterations = result.iterations)
    end
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    pair_count, model_count = size(scores)
    @variable(model, weights[1:length(programme_names), 1:model_count] >= 0)
    @constraint(model, [programme = 1:length(programme_names)], sum(weights[programme, model_index] for model_index = 1:model_count) == 1)
    @expression(model, ensemble_scores[i = 1:pair_count], sum(weights[programme_indices[i], model_index] * scores[i, model_index] for model_index = 1:model_count))
    level_weights = sample_weights(target, config)
    if config.objective == :weighted_mae
        @variable(model, absolute_errors[1:pair_count] >= 0)
        @constraint(model, [i = 1:pair_count], absolute_errors[i] >= ensemble_scores[i] - target[i])
        @constraint(model, [i = 1:pair_count], absolute_errors[i] >= target[i] - ensemble_scores[i])
        @objective(model, Min, sum(level_weights[i] * absolute_errors[i] for i = 1:pair_count))
    else
        @objective(model, Min, sum(level_weights[i] * (ensemble_scores[i] - target[i])^2 for i = 1:pair_count))
    end
    optimize!(model)
    status = termination_status(model)
    status == MOI.OPTIMAL || error("Optimization failed with status $status.")
    fitted_weights = value.(weights)
    fitted_scores = programme_ensemble_scores(scores, programme_indices, fitted_weights)
    return (weights = fitted_weights, programme_names = programme_names, fitted_scores = fitted_scores,
            objective_value = objective_value(model), status = string(status))
end

function index_programmes(programmes::AbstractVector, programme_names::AbstractVector{<:AbstractString})
    index_by_programme = Dict(name => index for (index, name) in enumerate(programme_names))
    return [get(index_by_programme, string(programme), 0) for programme in programmes]
end

function programme_ensemble_scores(scores::Matrix{Float64}, programme_indices::AbstractVector{<:Integer}, weights::Matrix{Float64})
    any(==(0), programme_indices) && error("A prediction programme is missing from the fitted weights.")
    return [sum(weights[programme_indices[i], model_index] * scores[i, model_index] for model_index = axes(scores, 2)) for i = axes(scores, 1)]
end

function fit_programme_calibration(raw_scores::AbstractVector{<:Real}, target::AbstractVector{<:Real}, programmes::AbstractVector)
    length(raw_scores) == length(target) || error("Raw scores and target values must have equal length.")
    length(programmes) == length(target) || error("Programme values and target values must have equal length.")
    programme_names = sort(unique(string.(programmes)))
    intercepts = Float64[]
    slopes = Float64[]
    for programme in programme_names
        programme_rows = findall(==(programme), string.(programmes))
        programme_scores = Float64.(raw_scores[programme_rows])
        programme_target = Float64.(target[programme_rows])
        score_mean = mean(programme_scores)
        target_mean = mean(programme_target)
        score_variance = sum((programme_scores .- score_mean) .^ 2)
        slope = score_variance == 0 ? 0.0 : max(0.0, sum((programme_scores .- score_mean) .* (programme_target .- target_mean)) / score_variance)
        push!(slopes, slope)
        push!(intercepts, target_mean - slope * score_mean)
    end
    return (programme_names = programme_names, intercepts = intercepts, slopes = slopes)
end

function apply_programme_calibration(raw_scores::AbstractVector{<:Real}, programmes::AbstractVector, calibration)
    length(raw_scores) == length(programmes) || error("Raw scores and programme values must have equal length.")
    programme_indices = index_programmes(programmes, calibration.programme_names)
    any(==(0), programme_indices) && error("A prediction programme is missing from the fitted calibration.")
    return [calibration.intercepts[programme_indices[i]] + calibration.slopes[programme_indices[i]] * raw_scores[i] for i in eachindex(raw_scores)]
end

function grouped_folds(groups::AbstractVector, fold_count::Integer; strata::Union{Nothing, AbstractVector} = nothing)
    fold_count >= 2 || error("fold_count must be at least 2.")
    unique_groups = sort(unique(string.(groups)))
    length(unique_groups) >= fold_count || error("fold_count exceeds the number of unique abstracts.")
    isnothing(strata) && return round_robin_grouped_folds(groups, unique_groups, fold_count)
    length(strata) == length(groups) || error("Strata and group values must have equal length.")
    stratum_names = sort(unique(string.(strata)))
    group_indices = Dict(group => index for (index, group) in enumerate(unique_groups))
    stratum_indices = Dict(stratum => index for (index, stratum) in enumerate(stratum_names))
    group_stratum_counts = zeros(Int, length(unique_groups), length(stratum_names))
    group_row_counts = zeros(Int, length(unique_groups))
    for (group, stratum) in zip(groups, strata)
        group_index = group_indices[string(group)]
        group_stratum_counts[group_index, stratum_indices[string(stratum)]] += 1
        group_row_counts[group_index] += 1
    end
    target_stratum_counts = vec(sum(group_stratum_counts, dims = 1)) ./ fold_count
    target_row_count = sum(group_row_counts) / fold_count
    fold_stratum_counts = zeros(Int, fold_count, length(stratum_names))
    fold_row_counts = zeros(Int, fold_count)
    fold_by_group = Dict{String, Int}()
    assignment_order = sort(collect(eachindex(unique_groups)); by = group_index ->
        (-maximum(group_stratum_counts[group_index, :]), -group_row_counts[group_index], unique_groups[group_index]))
    for group_index in assignment_order
        best_fold = first(1:fold_count)
        best_score = Inf
        for fold in 1:fold_count
            candidate_stratum_counts = copy(fold_stratum_counts)
            candidate_stratum_counts[fold, :] .+= group_stratum_counts[group_index, :]
            candidate_row_counts = copy(fold_row_counts)
            candidate_row_counts[fold] += group_row_counts[group_index]
            stratum_score = sum((candidate_stratum_counts .- target_stratum_counts').^2 ./ max.(target_stratum_counts', 1.0))
            row_score = sum((candidate_row_counts .- target_row_count).^2) / max(target_row_count, 1.0)
            score = stratum_score + row_score
            if score < best_score - eps() || (isapprox(score, best_score) && (fold_row_counts[fold], fold) < (fold_row_counts[best_fold], best_fold))
                best_fold = fold
                best_score = score
            end
        end
        fold_stratum_counts[best_fold, :] .+= group_stratum_counts[group_index, :]
        fold_row_counts[best_fold] += group_row_counts[group_index]
        fold_by_group[unique_groups[group_index]] = best_fold
    end
    return [fold_by_group[string(group)] for group in groups]
end

function round_robin_grouped_folds(groups::AbstractVector, unique_groups::AbstractVector{<:AbstractString}, fold_count::Integer)
    fold_by_group = Dict(group => mod(index - 1, fold_count) + 1 for (index, group) in enumerate(unique_groups))
    return [fold_by_group[string(group)] for group in groups]
end

function cross_validate(dataset; fold_count::Integer = 5, config::OptimizerConfig = OptimizerConfig())
    config.weight_scope in (:global, :programme) || error("Unsupported weight scope: $(config.weight_scope)")
    config.calibration_scope in (:none, :programme) || error("Unsupported calibration scope: $(config.calibration_scope)")
    (config.weight_scope == :programme || config.calibration_scope == :programme) && !(:Grouping in Symbol.(names(dataset.data))) && error("Programme-scoped weights or calibration require a Grouping column in the dataset.")
    folds = grouped_folds(dataset.data.Abstract_Index, fold_count; strata = dataset.data.RA2025_ID)
    prediction_frames = DataFrame[]
    weight_frames = DataFrame[]
    calibration_frames = DataFrame[]
    for fold in 1:fold_count
        train_indices = findall(!=(fold), folds)
        test_indices = findall(==(fold), folds)
        if config.weight_scope == :global
            result = solve_weights(dataset.scores[train_indices, :], dataset.target[train_indices], config)
            training_raw_scores = dataset.scores[train_indices, :] * result.weights
            test_raw_scores = dataset.scores[test_indices, :] * result.weights
            weight_frame = DataFrame(Fold = fill(fold, length(result.weights)), Weight_Scope = fill("global", length(result.weights)), Grouping = fill("All", length(result.weights)), Model_Slug = dataset.model_names, Weight = result.weights)
        else
            training_programmes = dataset.data.Grouping[train_indices]
            result = solve_weights(dataset.scores[train_indices, :], dataset.target[train_indices], training_programmes, config)
            training_programme_indices = index_programmes(training_programmes, result.programme_names)
            training_raw_scores = programme_ensemble_scores(dataset.scores[train_indices, :], training_programme_indices, result.weights)
            test_programme_indices = index_programmes(dataset.data.Grouping[test_indices], result.programme_names)
            test_raw_scores = programme_ensemble_scores(dataset.scores[test_indices, :], test_programme_indices, result.weights)
            weight_frame = DataFrame(Fold = repeat([fold], length(result.weights)), Weight_Scope = repeat(["programme"], length(result.weights)), Grouping = repeat(result.programme_names, inner = length(dataset.model_names)), Model_Slug = repeat(dataset.model_names, length(result.programme_names)), Weight = vec(result.weights'))
        end
        if config.calibration_scope == :programme
            calibration = fit_programme_calibration(training_raw_scores, dataset.target[train_indices], dataset.data.Grouping[train_indices])
            test_scores = apply_programme_calibration(test_raw_scores, dataset.data.Grouping[test_indices], calibration)
            push!(calibration_frames, DataFrame(Fold = fill(fold, length(calibration.programme_names)), Grouping = calibration.programme_names, Intercept = calibration.intercepts, Slope = calibration.slopes, Training_Pairs = fill(length(train_indices), length(calibration.programme_names))))
        else
            test_scores = test_raw_scores
        end
        predictions = hcat(dataset.data[test_indices, PAIR_KEYS], dataset.data[test_indices, Symbol.(dataset.model_names)])
        (config.weight_scope == :programme || config.calibration_scope == :programme) && (predictions.Grouping = dataset.data.Grouping[test_indices])
        predictions.Fold = fill(fold, length(test_indices))
        predictions.Evaluator_Affinity = dataset.target[test_indices]
        predictions.Raw_Weighted_Affinity = test_raw_scores
        predictions.Weighted_Affinity = test_scores
        predictions.Equal_Weight_Affinity = vec(mean(dataset.scores[test_indices, :], dims = 2))
        predictions.Expert_Level = affinity_level.(predictions.Evaluator_Affinity)
        predictions.Predicted_Level = affinity_level.(predictions.Weighted_Affinity)
        push!(prediction_frames, predictions)
        weight_frame.Train_Pairs = fill(length(train_indices), nrow(weight_frame))
        weight_frame.Test_Pairs = fill(length(test_indices), nrow(weight_frame))
        weight_frame.Objective_Value = fill(result.objective_value, nrow(weight_frame))
        weight_frame.Solver_Status = fill(result.status, nrow(weight_frame))
        push!(weight_frames, weight_frame)
    end
    predictions = vcat(prediction_frames...)
    calibrations = isempty(calibration_frames) ? DataFrame(Fold = Int[], Grouping = String[], Intercept = Float64[], Slope = Float64[], Training_Pairs = Int[]) : vcat(calibration_frames...)
    return (predictions = predictions, weights = vcat(weight_frames...), calibrations = calibrations,
            metrics = metrics_summary(predictions, config), fold_metrics = fold_metrics_summary(predictions, config))
end

function write_results(results, output_dir::AbstractString)
    mkpath(output_dir)
    CSV.write(joinpath(output_dir, "fold_weights.csv"), results.weights)
    nrow(results.calibrations) > 0 && CSV.write(joinpath(output_dir, "fold_calibrations.csv"), results.calibrations)
    CSV.write(joinpath(output_dir, "out_of_fold_predictions.csv"), results.predictions)
    CSV.write(joinpath(output_dir, "metrics_summary.csv"), results.metrics)
    CSV.write(joinpath(output_dir, "fold_metrics_summary.csv"), results.fold_metrics)
end