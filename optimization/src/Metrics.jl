function affinity_level(score::Real)
    score <= 40 && return "Low"
    score <= 80 && return "Moderate"
    return "High"
end

function sample_weights(target::AbstractVector, config::OptimizerConfig)
    multipliers = (config.low_multiplier, config.moderate_multiplier, config.high_multiplier)
    all(multipliers .>= 0) || error("All level multipliers must be non-negative.")
    any(multipliers .> 0) || error("At least one level multiplier must be positive.")
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

function metrics_summary(predictions::DataFrame, config::OptimizerConfig)
    expert_scores = Float64.(predictions.Evaluator_Affinity)
    level_weights = sample_weights(expert_scores, config)
    approaches = [(:Optimized, Float64.(predictions.Weighted_Affinity)), (:Equal_Weight, Float64.(predictions.Equal_Weight_Affinity))]
    rows = NamedTuple[]
    for (name, scores) in approaches
        errors = scores .- expert_scores
        push!(rows, (Approach = String(name), Metric = "Weighted_MAE", Value = sum(level_weights .* abs.(errors)) / sum(level_weights)))
        push!(rows, (Approach = String(name), Metric = "Weighted_MSE", Value = sum(level_weights .* errors.^2) / sum(level_weights)))
        push!(rows, (Approach = String(name), Metric = "MAE", Value = sum(abs.(errors)) / length(errors)))
        push!(rows, (Approach = String(name), Metric = "MSE", Value = sum(errors.^2) / length(errors)))
        push!(rows, (Approach = String(name), Metric = "Mean_Signed_Error", Value = sum(errors) / length(errors)))
        predicted_levels = affinity_level.(scores)
        push!(rows, (Approach = String(name), Metric = "Three_Level_Accuracy", Value = sum(predicted_levels .== predictions.Expert_Level) / length(scores)))
        append!(rows, [(Approach = String(name), metric.Metric, metric.Value)
                       for metric in classification_metrics(predictions.Expert_Level, predicted_levels)])
    end
    return DataFrame(rows)
end

function fold_metrics_summary(predictions::DataFrame, config::OptimizerConfig)
    :Fold in Symbol.(names(predictions)) || error("Predictions must include a Fold column.")
    fold_frames = DataFrame[]
    for fold in sort(unique(predictions.Fold))
        fold_metrics = metrics_summary(filter(:Fold => ==(fold), predictions), config)
        insertcols!(fold_metrics, 1, :Fold => fill(string(fold), nrow(fold_metrics)))
        push!(fold_frames, fold_metrics)
    end
    fold_metrics = vcat(fold_frames...)
    averages = combine(groupby(fold_metrics, [:Approach, :Metric]), :Value => mean => :Value)
    insertcols!(averages, 1, :Fold => fill("Average", nrow(averages)))
    return vcat(fold_metrics, averages)
end