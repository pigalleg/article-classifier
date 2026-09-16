module LLMEnsembleOptimizer

using CSV
using DataFrames
using Gurobi
using JuMP
using Statistics
import MathOptInterface as MOI

include("Optimization.jl")
include("Metrics.jl")
include("GradientDescent.jl")

export OptimizerConfig, affinity_level, apply_programme_calibration, cross_validate, descend_weights,
       fit_programme_calibration, fold_metrics_summary, grouped_folds, load_dataset, metrics_summary,
       project_onto_simplex!, sample_weights, soft_kappa_loss_and_gradient, solve_weights, write_results

end
