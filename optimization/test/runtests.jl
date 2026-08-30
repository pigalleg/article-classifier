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
