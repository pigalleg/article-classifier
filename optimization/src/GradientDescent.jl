const MAXIMUM_BACKTRACKING_STEPS = 30
const STEP_GROWTH_FACTOR = 1.5
const SPARSIFICATION_THRESHOLD = 1e-4

logistic(value::Float64) = 1.0 / (1.0 + exp(-value))

function project_onto_simplex!(weights::AbstractVector{Float64})
    model_count = length(weights)
    sorted_weights = sort(weights; rev = true)
    cumulative_weight = 0.0
    threshold = 0.0
    for index in 1:model_count
        cumulative_weight += sorted_weights[index]
        if sorted_weights[index] * index > cumulative_weight - 1.0
            threshold = (cumulative_weight - 1.0) / index
        end
    end
    @. weights = max(weights - threshold, 0.0)
    return weights
end

function project_rows_onto_simplex!(weights::Matrix{Float64})
    for programme in axes(weights, 1)
        project_onto_simplex!(view(weights, programme, :))
    end
    return weights
end

function ensemble_scores_from_weights(scores::Matrix{Float64}, programme_indices::AbstractVector{<:Integer}, weights::Matrix{Float64})
    return [sum(weights[programme_indices[i], m] * scores[i, m] for m in axes(scores, 2)) for i in axes(scores, 1)]
end

function expert_level_indices(target::AbstractVector{Float64})
    return [score <= 40 ? 1 : score <= 80 ? 2 : 3 for score in target]
end

function quadratic_disagreement_weights()
    level_count = length(AFFINITY_LEVELS)
    return [(actual - predicted)^2 / (level_count - 1)^2 for actual in 1:level_count, predicted in 1:level_count]
end

function score_error_loss_and_gradient(ensemble_scores::Vector{Float64}, target::Vector{Float64},
                                       sample_multipliers::Vector{Float64}, objective::Symbol)
    multiplier_total = sum(sample_multipliers)
    score_gradient = similar(ensemble_scores)
    loss = 0.0
    for i in eachindex(ensemble_scores)
        error_value = ensemble_scores[i] - target[i]
        if objective == :weighted_mae
            loss += sample_multipliers[i] * abs(error_value)
            score_gradient[i] = sample_multipliers[i] * sign(error_value) / multiplier_total
        else
            loss += sample_multipliers[i] * error_value^2
            score_gradient[i] = 2.0 * sample_multipliers[i] * error_value / multiplier_total
        end
    end
    return loss / multiplier_total, score_gradient
end

function soft_kappa_loss_and_gradient(ensemble_scores::Vector{Float64}, expert_levels::Vector{Int},
                                      sample_multipliers::Vector{Float64}, temperature::Float64)
    temperature > 0 || error("The level temperature must be positive.")
    level_count = length(AFFINITY_LEVELS)
    disagreement_weights = quadratic_disagreement_weights()
    pair_count = length(ensemble_scores)
    multiplier_total = sum(sample_multipliers)
    low_gates = [logistic((ensemble_scores[i] - 40.0) / temperature) for i in 1:pair_count]
    high_gates = [logistic((ensemble_scores[i] - 80.0) / temperature) for i in 1:pair_count]
    level_probabilities = hcat(1.0 .- low_gates, low_gates .- high_gates, high_gates)
    expert_totals = zeros(level_count)
    predicted_totals = zeros(level_count)
    for i in 1:pair_count
        expert_totals[expert_levels[i]] += sample_multipliers[i]
        for level in 1:level_count
            predicted_totals[level] += sample_multipliers[i] * level_probabilities[i, level]
        end
    end
    observed_disagreement = sum(sample_multipliers[i] * disagreement_weights[expert_levels[i], level] * level_probabilities[i, level]
                                for i in 1:pair_count, level in 1:level_count)
    expected_disagreement = sum(disagreement_weights[actual, predicted] * expert_totals[actual] * predicted_totals[predicted]
                                for actual in 1:level_count, predicted in 1:level_count) / multiplier_total
    expected_disagreement > 0 || error("Soft kappa is undefined when the expected disagreement is zero.")
    marginal_weights = [sum(disagreement_weights[actual, predicted] * expert_totals[actual] for actual in 1:level_count) / multiplier_total
                        for predicted in 1:level_count]
    score_gradient = similar(ensemble_scores)
    for i in 1:pair_count
        low_gate_slope = low_gates[i] * (1.0 - low_gates[i]) / temperature
        high_gate_slope = high_gates[i] * (1.0 - high_gates[i]) / temperature
        probability_slopes = (-low_gate_slope, low_gate_slope - high_gate_slope, high_gate_slope)
        gradient_value = 0.0
        for level in 1:level_count
            probability_gradient = sample_multipliers[i] *
                (disagreement_weights[expert_levels[i], level] * expected_disagreement - observed_disagreement * marginal_weights[level]) /
                expected_disagreement^2
            gradient_value += probability_gradient * probability_slopes[level]
        end
        score_gradient[i] = gradient_value
    end
    return observed_disagreement / expected_disagreement, score_gradient
end

function loss_and_score_gradient(ensemble_scores::Vector{Float64}, target::Vector{Float64}, expert_levels::Vector{Int},
                                 sample_multipliers::Vector{Float64}, config::OptimizerConfig)
    config.objective == :soft_qwk &&
        return soft_kappa_loss_and_gradient(ensemble_scores, expert_levels, sample_multipliers, config.level_temperature)
    return score_error_loss_and_gradient(ensemble_scores, target, sample_multipliers, config.objective)
end

function starting_weight_matrices(scores::Matrix{Float64}, target::Vector{Float64}, expert_levels::Vector{Int},
                                  sample_multipliers::Vector{Float64}, programme_indices::AbstractVector{<:Integer},
                                  programme_count::Integer, config::OptimizerConfig)
    model_count = size(scores, 2)
    uniform_start = fill(1.0 / model_count, programme_count, model_count)
    config.restart_count <= 1 && return [uniform_start]
    single_model_losses = map(1:model_count) do model_index
        vertex = zeros(programme_count, model_count)
        vertex[:, model_index] .= 1.0
        first(loss_and_score_gradient(ensemble_scores_from_weights(scores, programme_indices, vertex), target, expert_levels, sample_multipliers, config))
    end
    ranked_models = sortperm(single_model_losses)
    starts = [uniform_start]
    for model_index in ranked_models[1:min(config.restart_count - 1, model_count)]
        vertex = zeros(programme_count, model_count)
        vertex[:, model_index] .= 1.0
        push!(starts, vertex)
    end
    return starts
end

function normalized_descent_direction(weight_gradient::Matrix{Float64})
    largest_gradient = maximum(abs, weight_gradient)
    largest_gradient == 0 && return nothing
    return weight_gradient ./ largest_gradient
end

function weight_gradient_from_scores(scores::Matrix{Float64}, programme_indices::AbstractVector{<:Integer},
                                     score_gradient::Vector{Float64}, programme_count::Integer)
    weight_gradient = zeros(programme_count, size(scores, 2))
    for i in axes(scores, 1)
        programme = programme_indices[i]
        for model_index in axes(scores, 2)
            weight_gradient[programme, model_index] += score_gradient[i] * scores[i, model_index]
        end
    end
    return weight_gradient
end

function sparsified_weights(weights::Matrix{Float64})
    sparse_weights = [weight < SPARSIFICATION_THRESHOLD ? 0.0 : weight for weight in weights]
    for programme in axes(sparse_weights, 1)
        row_total = sum(view(sparse_weights, programme, :))
        row_total > 0 || return copy(weights)
        sparse_weights[programme, :] ./= row_total
    end
    return sparse_weights
end

function descend_from_start(starting_weights::Matrix{Float64}, scores::Matrix{Float64}, target::Vector{Float64},
                            expert_levels::Vector{Int}, sample_multipliers::Vector{Float64},
                            programme_indices::AbstractVector{<:Integer}, programme_count::Integer, config::OptimizerConfig)
    evaluate(weights) = loss_and_score_gradient(ensemble_scores_from_weights(scores, programme_indices, weights),
                                                target, expert_levels, sample_multipliers, config)
    weights = copy(starting_weights)
    loss, score_gradient = evaluate(weights)
    step_size = config.learning_rate
    negligible_improvements = 0
    iterations = 0
    converged = false
    for _ in 1:config.max_iterations
        iterations += 1
        direction = normalized_descent_direction(weight_gradient_from_scores(scores, programme_indices, score_gradient, programme_count))
        if isnothing(direction)
            converged = true
            break
        end
        step_size = min(step_size * STEP_GROWTH_FACTOR, config.learning_rate)
        accepted = false
        for _ in 1:MAXIMUM_BACKTRACKING_STEPS
            candidate_weights = project_rows_onto_simplex!(weights .- step_size .* direction)
            candidate_loss, candidate_score_gradient = evaluate(candidate_weights)
            if candidate_loss < loss
                improvement = (loss - candidate_loss) / max(abs(loss), 1e-12)
                negligible_improvements = improvement < config.tolerance ? negligible_improvements + 1 : 0
                weights = candidate_weights
                loss = candidate_loss
                score_gradient = candidate_score_gradient
                accepted = true
                break
            end
            step_size /= 2
        end
        if !accepted || negligible_improvements >= config.patience
            converged = true
            break
        end
    end
    candidate_weights = sparsified_weights(weights)
    candidate_loss, _ = evaluate(candidate_weights)
    candidate_loss <= loss && ((weights, loss) = (candidate_weights, candidate_loss))
    return (weights = weights, loss = loss, iterations = iterations, converged = converged)
end

function descend_weights(scores::Matrix{Float64}, target::Vector{Float64}, programme_indices::AbstractVector{<:Integer},
                         programme_count::Integer, config::OptimizerConfig)
    config.learning_rate > 0 || error("The gradient learning rate must be positive.")
    config.max_iterations >= 1 || error("The gradient iteration budget must be at least one.")
    config.patience >= 1 || error("The gradient patience must be at least one.")
    config.restart_count >= 1 || error("The gradient restart count must be at least one.")
    sample_multipliers = sample_weights(target, config)
    expert_levels = expert_level_indices(target)
    best_weights = fill(1.0 / size(scores, 2), programme_count, size(scores, 2))
    best_loss = Inf
    total_iterations = 0
    converged = false
    for starting_weights in starting_weight_matrices(scores, target, expert_levels, sample_multipliers, programme_indices, programme_count, config)
        result = descend_from_start(starting_weights, scores, target, expert_levels, sample_multipliers,
                                    programme_indices, programme_count, config)
        total_iterations += result.iterations
        if result.loss < best_loss
            best_loss = result.loss
            best_weights = result.weights
            converged = result.converged
        end
    end
    scale = config.objective == :soft_qwk ? 1.0 : sum(sample_multipliers)
    status = converged ? "GRADIENT_CONVERGED" : "GRADIENT_ITERATION_LIMIT"
    return (weights = best_weights, objective_value = best_loss * scale, status = status, iterations = total_iterations)
end
