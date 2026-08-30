using LLMEnsembleOptimizer
using CSV
using DataFrames

function argument_value(arguments, name, default = nothing)
    index = findfirst(==(name), arguments)
    isnothing(index) && return default
    index < length(arguments) || error("Missing value for $name.")
    return arguments[index + 1]
end

repo_root = normpath(joinpath(@__DIR__, ".."))
expert_run = argument_value(ARGS, "--expert-run", "20260618_Mark_batch1_v4")
benchmark_run = argument_value(ARGS, "--benchmark-run", "20260724_155639")
objective = Symbol(argument_value(ARGS, "--objective", "weighted_mae"))
weight_scope = Symbol(argument_value(ARGS, "--weight-scope", "global"))
calibration_scope = Symbol(argument_value(ARGS, "--calibration-scope", "none"))
low_multiplier = parse(Float64, argument_value(ARGS, "--low-multiplier", "1.0"))
moderate_multiplier = parse(Float64, argument_value(ARGS, "--moderate-multiplier", "2.0"))
high_multiplier = parse(Float64, argument_value(ARGS, "--high-multiplier", "2.0"))
fold_count = parse(Int, argument_value(ARGS, "--folds", "5"))
output_dir = argument_value(ARGS, "--output-dir", joinpath(repo_root, "data", "results", "ensemble_weight_optimizer"))
fit_all = "--fit-all" in ARGS

expert_path = joinpath(repo_root, "data", "processed", "affinity_calibration", expert_run, "calibration_abstract_question_affinities.csv")
benchmark_path = joinpath(repo_root, "data", "results", "affinity_benchmark", benchmark_run, "merged_ra_affinities.csv")
programme_mapping_path = joinpath(repo_root, "data", "processed", "ra_grouping_ra2025.csv")
dataset = load_dataset(expert_path, benchmark_path; programme_mapping_path = programme_mapping_path)
config = OptimizerConfig(objective = objective, weight_scope = weight_scope, low_multiplier = low_multiplier,
    calibration_scope = calibration_scope, moderate_multiplier = moderate_multiplier, high_multiplier = high_multiplier)
results = cross_validate(dataset; fold_count = fold_count, config = config)
write_results(results, output_dir)
if fit_all
    if config.weight_scope == :global
        final_result = solve_weights(dataset.scores, dataset.target, config)
        final_weights = DataFrame(Weight_Scope = fill("global", length(final_result.weights)), Grouping = fill("All", length(final_result.weights)),
            Model_Slug = dataset.model_names, Weight = final_result.weights)
    else
        final_result = solve_weights(dataset.scores, dataset.target, dataset.data.Grouping, config)
        final_weights = DataFrame(Weight_Scope = repeat(["programme"], length(final_result.weights)),
            Grouping = repeat(final_result.programme_names, inner = length(dataset.model_names)),
            Model_Slug = repeat(dataset.model_names, length(final_result.programme_names)), Weight = vec(final_result.weights'))
    end
    final_weights.Training_Pairs = fill(dataset.retained_rows, nrow(final_weights))
    final_weights.Objective_Value = fill(final_result.objective_value, nrow(final_weights))
    final_weights.Solver_Status = fill(final_result.status, nrow(final_weights))
    CSV.write(joinpath(output_dir, "final_weights.csv"), final_weights)
    if config.calibration_scope == :programme
        final_calibration = fit_programme_calibration(final_result.fitted_scores, dataset.target, dataset.data.Grouping)
        CSV.write(joinpath(output_dir, "final_calibration.csv"), DataFrame(Calibration_Scope = fill("programme", length(final_calibration.programme_names)),
            Grouping = final_calibration.programme_names, Intercept = final_calibration.intercepts, Slope = final_calibration.slopes,
            Training_Pairs = fill(dataset.retained_rows, length(final_calibration.programme_names))))
    end
end
open(joinpath(output_dir, "run_metadata.toml"), "w") do io
    println(io, "expert_run = \"$expert_run\"")
    println(io, "benchmark_run = \"$benchmark_run\"")
    println(io, "objective = \"$objective\"")
    println(io, "weight_scope = \"$weight_scope\"")
    println(io, "calibration_scope = \"$calibration_scope\"")
    println(io, "low_multiplier = $low_multiplier")
    println(io, "moderate_multiplier = $moderate_multiplier")
    println(io, "high_multiplier = $high_multiplier")
    println(io, "fold_count = $fold_count")
    println(io, "joined_rows = $(dataset.joined_rows)")
    println(io, "retained_rows = $(dataset.retained_rows)")
    println(io, "excluded_models = [$(join(repr.(dataset.excluded_models), ", "))]")
    println(io, "fit_all = $fit_all")
end
println("Wrote $(length(results.predictions.Fold)) out-of-fold predictions to $output_dir")
