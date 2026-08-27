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

@testset "grouped folds" begin
    groups = ["a", "a", "b", "b", "c", "d"]
    folds = grouped_folds(groups, 2)
    @test folds[1] == folds[2]
    @test folds[3] == folds[4]
    @test length(unique(folds)) == 2
end

@testset "affinity levels" begin
    @test affinity_level(40) == "Low"
    @test affinity_level(40.1) == "Moderate"
    @test affinity_level(80) == "Moderate"
    @test affinity_level(80.1) == "High"
    @test sample_weights([40.0, 50.0, 81.0], OptimizerConfig(low_multiplier = 1.5,
        moderate_multiplier = 1.0, high_multiplier = 4.0)) == [1.5, 1.0, 4.0]
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
end
