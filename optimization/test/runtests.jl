using Test
using DataFrames
using LLMEnsembleOptimizer

@testset "weighted MAE optimizer" begin
    scores = [10.0 80.0; 20.0 70.0; 90.0 20.0; 100.0 10.0]
    target = [10.0, 20.0, 90.0, 100.0]
    try
        result = solve_weights(scores, target)
        @test all(result.weights .>= -1e-8)
        @test isapprox(sum(result.weights), 1.0; atol = 1e-8)
        @test result.weights[1] > result.weights[2]
    catch error
        occursin("No Gurobi license found", sprint(showerror, error)) || rethrow()
        @test_skip "Gurobi license is unavailable in this environment."
    end
end

@testset "weighted MSE optimizer" begin
    scores = [10.0 80.0; 20.0 70.0; 90.0 20.0; 100.0 10.0]
    target = [10.0, 20.0, 90.0, 100.0]
    try
        result = solve_weights(scores, target, OptimizerConfig(objective = :weighted_mse))
        @test all(result.weights .>= -1e-8)
        @test isapprox(sum(result.weights), 1.0; atol = 1e-8)
        @test result.weights[1] > result.weights[2]
    catch error
        occursin("No Gurobi license found", sprint(showerror, error)) || rethrow()
        @test_skip "Gurobi license is unavailable in this environment."
    end
end

@testset "programme-scoped optimizer" begin
    scores = [10.0 80.0; 20.0 70.0; 90.0 20.0; 100.0 10.0]
    target = [10.0, 20.0, 20.0, 10.0]
    programmes = ["A", "A", "B", "B"]
    config = OptimizerConfig(weight_scope = :programme)
    try
        result = solve_weights(scores, target, programmes, config)
        @test result.programme_names == ["A", "B"]
        @test all(result.weights .>= -1e-8)
        @test all(isapprox.(vec(sum(result.weights, dims = 2)), 1.0; atol = 1e-8))
        @test result.weights[1, 1] > result.weights[1, 2]
        @test result.weights[2, 2] > result.weights[2, 1]
    catch error
        occursin("No Gurobi license found", sprint(showerror, error)) || rethrow()
        @test_skip "Gurobi license is unavailable in this environment."
    end
end

@testset "programme affine calibration" begin
    raw_scores = [10.0, 20.0, 30.0, 40.0]
    target = [25.0, 45.0, 65.0, 85.0]
    programmes = ["A", "A", "B", "B"]
    calibration = fit_programme_calibration(raw_scores, target, programmes)
    @test calibration.programme_names == ["A", "B"]
    @test calibration.intercepts == [5.0, 5.0]
    @test calibration.slopes == [2.0, 2.0]
    @test apply_programme_calibration([15.0, 35.0], ["A", "B"], calibration) == [35.0, 75.0]
end

@testset "grouped folds" begin
    groups = ["a", "a", "b", "b", "c", "d"]
    folds = grouped_folds(groups, 2)
    @test folds[1] == folds[2]
    @test folds[3] == folds[4]
    @test length(unique(folds)) == 2
end

@testset "question-stratified grouped folds" begin
    groups = ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"]
    questions = ["q1", "q2", "q1", "q2", "q1", "q2", "q1", "q2", "q1", "q2", "q1", "q2"]
    folds = grouped_folds(groups, 3; strata = questions)
    @test all(folds[2index - 1] == folds[2index] for index in 1:6)
    for question in unique(questions)
        question_counts = [count((questions .== question) .& (folds .== fold)) for fold in 1:3]
        @test maximum(question_counts) - minimum(question_counts) <= 1
    end
end

@testset "affinity levels" begin
    @test affinity_level(40) == "Low"
    @test affinity_level(40.1) == "Moderate"
    @test affinity_level(80) == "Moderate"
    @test affinity_level(80.1) == "High"
    @test sample_weights([40.0, 50.0, 81.0], OptimizerConfig(low_multiplier = 1.5,
        moderate_multiplier = 1.0, high_multiplier = 4.0)) == [1.5, 1.0, 4.0]
    @test sample_weights([40.0, 50.0, 81.0], OptimizerConfig(low_multiplier = 0.0,
        moderate_multiplier = 1.0, high_multiplier = 0.0)) == [0.0, 1.0, 0.0]
    @test_throws ErrorException sample_weights([40.0], OptimizerConfig(low_multiplier = 0.0,
        moderate_multiplier = 0.0, high_multiplier = 0.0))
end

@testset "classification metrics summary" begin
    predictions = DataFrame(
        Evaluator_Affinity = [10.0, 20.0, 50.0, 90.0],
        Weighted_Affinity = [10.0, 50.0, 50.0, 90.0],
        Equal_Weight_Affinity = [10.0, 20.0, 50.0, 90.0],
        Expert_Level = ["Low", "Low", "Moderate", "High"],
    )
    metrics = metrics_summary(predictions, OptimizerConfig())
    optimized = filter(:Approach => ==("Optimized"), metrics)
    metric_value(name) = only(optimized.Value[optimized.Metric .== name])
    @test metric_value("Quadratic_Weighted_Kappa") == 0.8
    @test metric_value("Low_Support") == 2.0
    @test metric_value("Low_Precision") == 1.0
    @test metric_value("Low_Recall") == 0.5
    @test metric_value("Low_One_vs_Rest_Accuracy") == 0.75
    @test metric_value("Low_True_Positive") == 1.0
    @test metric_value("Low_False_Negative") == 1.0
    @test metric_value("Weighted_MSE") == 150.0
    @test metric_value("MSE") == 225.0
end

@testset "fold metrics summary" begin
    predictions = DataFrame(
        Fold = [1, 1, 2, 2],
        Evaluator_Affinity = [10.0, 20.0, 50.0, 90.0],
        Weighted_Affinity = [10.0, 50.0, 50.0, 90.0],
        Equal_Weight_Affinity = [10.0, 20.0, 50.0, 90.0],
        Expert_Level = ["Low", "Low", "Moderate", "High"],
    )
    metrics = fold_metrics_summary(predictions, OptimizerConfig())
    @test Set(metrics.Fold) == Set(["1", "2", "Average"])
    average = filter(:Fold => ==("Average"), metrics)
    optimized = filter(:Approach => ==("Optimized"), average)
    @test only(optimized.Value[optimized.Metric .== "MSE"]) == 225.0
end

@testset "classification metrics with no predicted class" begin
    predictions = DataFrame(
        Evaluator_Affinity = [10.0, 90.0],
        Weighted_Affinity = [10.0, 50.0],
        Equal_Weight_Affinity = [10.0, 50.0],
        Expert_Level = ["Low", "High"],
    )
    metrics = metrics_summary(predictions, OptimizerConfig())
    optimized = filter(:Approach => ==("Optimized"), metrics)
    @test only(optimized.Value[optimized.Metric .== "High_Precision"]) == 0.0
end

@testset "simplex projection" begin
    @test project_onto_simplex!([0.5, 0.5]) == [0.5, 0.5]
    @test isapprox(sum(project_onto_simplex!([3.0, -1.0, 0.2])), 1.0; atol = 1e-12)
    @test all(project_onto_simplex!([3.0, -1.0, 0.2]) .>= 0.0)
    @test project_onto_simplex!([10.0, 0.0, 0.0]) == [1.0, 0.0, 0.0]
    @test isapprox(project_onto_simplex!([0.1, 0.1, 0.1]), fill(1 / 3, 3); atol = 1e-12)
end

@testset "soft kappa gradient matches finite differences" begin
    ensemble_scores = [12.0, 38.0, 55.0, 79.0, 95.0, 41.0]
    expert_levels = [1, 1, 2, 2, 3, 2]
    sample_multipliers = [1.0, 1.0, 2.0, 2.0, 4.0, 2.0]
    loss, gradient = soft_kappa_loss_and_gradient(ensemble_scores, expert_levels, sample_multipliers, 5.0)
    step = 1e-6
    for index in eachindex(ensemble_scores)
        forward_scores = copy(ensemble_scores)
        backward_scores = copy(ensemble_scores)
        forward_scores[index] += step
        backward_scores[index] -= step
        forward_loss, _ = soft_kappa_loss_and_gradient(forward_scores, expert_levels, sample_multipliers, 5.0)
        backward_loss, _ = soft_kappa_loss_and_gradient(backward_scores, expert_levels, sample_multipliers, 5.0)
        @test isapprox(gradient[index], (forward_loss - backward_loss) / (2step); atol = 1e-6)
    end
    @test 0.0 <= loss <= 2.0
end

@testset "gradient descent recovers the weighted MAE optimum" begin
    scores = [10.0 80.0; 20.0 70.0; 90.0 20.0; 100.0 10.0]
    target = [10.0, 20.0, 90.0, 100.0]
    config = OptimizerConfig(solver = :gradient, max_iterations = 4000)
    result = solve_weights(scores, target, config)
    @test all(result.weights .>= -1e-8)
    @test isapprox(sum(result.weights), 1.0; atol = 1e-8)
    @test isapprox(result.weights[1], 1.0; atol = 1e-3)
    @test result.objective_value < 1e-2
end

@testset "gradient descent agrees with the exact solver" begin
    scores = [12.0 70.0 30.0; 25.0 65.0 44.0; 88.0 25.0 61.0; 96.0 12.0 77.0; 51.0 48.0 55.0]
    target = [15.0, 30.0, 85.0, 95.0, 50.0]
    for objective in (:weighted_mae, :weighted_mse)
        gradient_result = solve_weights(scores, target, OptimizerConfig(objective = objective, solver = :gradient, max_iterations = 6000))
        try
            exact_result = solve_weights(scores, target, OptimizerConfig(objective = objective))
            @test gradient_result.objective_value <= exact_result.objective_value * 1.02 + 1e-6
        catch error
            occursin("No Gurobi license found", sprint(showerror, error)) || rethrow()
            @test_skip "Gurobi license is unavailable in this environment."
        end
    end
end

@testset "programme-scoped gradient descent" begin
    scores = [10.0 80.0; 20.0 70.0; 90.0 20.0; 100.0 10.0]
    target = [10.0, 20.0, 20.0, 10.0]
    programmes = ["A", "A", "B", "B"]
    config = OptimizerConfig(weight_scope = :programme, solver = :gradient, max_iterations = 4000)
    result = solve_weights(scores, target, programmes, config)
    @test result.programme_names == ["A", "B"]
    @test all(result.weights .>= -1e-8)
    @test all(isapprox.(vec(sum(result.weights, dims = 2)), 1.0; atol = 1e-8))
    @test result.weights[1, 1] > result.weights[1, 2]
    @test result.weights[2, 2] > result.weights[2, 1]
end

@testset "soft kappa objective beats score-error fitting on level agreement" begin
    accurate_model = [5.0, 15.0, 45.0, 60.0, 85.0, 110.0, 30.0, 95.0]
    biased_model = [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 41.0, 44.0]
    scores = hcat(accurate_model, biased_model)
    target = [5.0, 15.0, 45.0, 60.0, 85.0, 110.0, 30.0, 95.0]
    kappa_result = solve_weights(scores, target, OptimizerConfig(objective = :soft_qwk, solver = :gradient,
        low_multiplier = 1.0, moderate_multiplier = 1.0, high_multiplier = 1.0, max_iterations = 6000, restart_count = 3))
    @test kappa_result.weights[1] > kappa_result.weights[2]
    predicted_levels = affinity_level.(kappa_result.fitted_scores)
    @test predicted_levels == affinity_level.(target)
    @test kappa_result.objective_value < 0.5
end

@testset "solver and objective validation" begin
    scores = [10.0 80.0; 20.0 70.0]
    target = [10.0, 20.0]
    @test_throws ErrorException solve_weights(scores, target, OptimizerConfig(objective = :soft_qwk))
    @test_throws ErrorException solve_weights(scores, target, OptimizerConfig(objective = :hinge, solver = :gradient))
    @test_throws ErrorException solve_weights(scores, target, OptimizerConfig(solver = :adam))
end

@testset "gradient cross-validation reports solver status" begin
    dataset = (
        data = DataFrame(
            Abstract_Index = [1, 1, 2, 2, 3, 3, 4, 4],
            RA2025_ID = ["q1", "q2", "q1", "q2", "q1", "q2", "q1", "q2"],
            Grouping = fill("IBR", 8),
            model_a = [10.0, 30.0, 50.0, 70.0, 90.0, 20.0, 40.0, 100.0],
            model_b = [80.0, 60.0, 45.0, 30.0, 15.0, 70.0, 55.0, 5.0],
        ),
        target = [10.0, 30.0, 50.0, 70.0, 90.0, 20.0, 40.0, 100.0],
        scores = [10.0 80.0; 30.0 60.0; 50.0 45.0; 70.0 30.0; 90.0 15.0; 20.0 70.0; 40.0 55.0; 100.0 5.0],
        model_names = ["model_a", "model_b"],
    )
    results = cross_validate(dataset; fold_count = 2, config = OptimizerConfig(objective = :soft_qwk, solver = :gradient,
        low_multiplier = 1.0, moderate_multiplier = 1.0, high_multiplier = 1.0, max_iterations = 2000))
    @test nrow(results.predictions) == 8
    @test all(startswith.(results.weights.Solver_Status, "GRADIENT_"))
    @test all(isapprox.(combine(groupby(results.weights, :Fold), :Weight => sum => :Total).Total, 1.0; atol = 1e-8))
end
