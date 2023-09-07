module SVMExperiment

using MLJ, MLJBase

import DrWatson: allaccess, default_prefix, default_allowed, produce_or_load
using Dates
using DrWatson
import ..ExecutionInfo, ..ExperimentParameters, ..default_resampling, ..default_measure

import ...Models

using MLJ, MLJBase
using LIBSVM

import MLJ: fitted_params, predict

import ...DataSets
import ...DataSets: DataSet, CategoricalDataSet, RegressionDataSet, MNIST, DelveRegressionDataSet, is_regression

import ...Utils

export SVMConfig

"""
    SVMConfig <: ExperimentParameters

The configuration for running an SVM experiment.
"""
@kwdef struct SVMConfig <: ExperimentParameters
    folder::String = "svms"
    dataset::DataSet

    # Evaluation parameters
    resampling::ResamplingStrategy = default_resampling(dataset)
    measure::MLJBase.Measure = default_measure(dataset)

    # Model parameters
    kernel::LIBSVM.Kernel.KERNEL = LIBSVM.Kernel.RadialBasis
    cost::Float64 = 1.0
    gamma::Float64 = 0.0

    scale_sigma::Bool = false

    subsample::Union{Float64,Nothing} = nothing

    # Regression specific
    epsilon::Union{Float64,Nothing} = 0.1

    extra_params::Dict{Symbol,Any} = Dict{Symbol,Any}(:max_iter => Int32(1e5))
end

is_regression(svm::SVMConfig) = svm.dataset |> is_regression

function model_parameters(svm::SVMConfig)
    common = [:kernel, :cost, :gamma]

    if is_regression(svm)
        return [common; :epsilon]
    end
    return common
end

# savename configuration
allaccess(svm::SVMConfig) = [:dataset, :resampling, :subsample, :measure, model_parameters(svm)...]

default_prefix(svm::SVMConfig) = is_regression(svm) ? "SVR" : "SVC"

default_savefile(svm::SVMConfig) = datadir(svm.folder, savename(svm, "jld2"))

function model(svm::SVMConfig, gamma=nothing)
    parameters = map(model_parameters(svm)) do param
        # If we are scaling sigma, we use the value given to us
        if param == :gamma && svm.scale_sigma
            (param, gamma)
        else
            (param, getfield(svm, param))
        end
    end

    Models.pipeline(svm.dataset; parameters..., svm.extra_params...)
end

function run(svm::SVMConfig)::Tuple{PerformanceEvaluation,ExecutionInfo,Machine}
    start = Dates.now()

    (Xtrain, Xtest), (ytrain, ytest) = if isnothing(svm.subsample)
        partition(svm.dataset)
    else
        # If we are sub-sampling, we discard a fraction of the training data
        (X, _), (y, _) = partition(unpack(svm.dataset), svm.subsample; shuffle=true, rng=45671, multi=true)
        partition((X, y), 0.8, shuffle=true, rng=1234, multi=true)
    end

    n = length(ytrain)
    if svm.scale_sigma && svm.kernel != LIBSVM.Kernel.RadialBasis
        sigma = Utils.gamma2sigma(svm.gamma) / n
        gamma = Utils.sigma2gamma(sigma)
    end

    pipe = svm.scale_sigma ? model(svm, gamma) : model(svm)
    if svm.kernel in [LIBSVM.Kernel.Acos0, LIBSVM.Kernel.Acos1, LIBSVM.Kernel.Acos2]
        pipe.multiplier.factor = sqrt(Utils.gamma2sigma(svm.gamma))
    end
    mach = machine(pipe, Xtrain, ytrain, cache=false)

    result = evaluate!(mach; svm.resampling, svm.measure)

    yhat = MLJ.predict(mach, Xtest)
    measure_test = svm.measure(yhat, ytest)

    duration = Dates.now() - start

    info = ExecutionInfo(; duration, measure_test, sigma_scaled=svm.scale_sigma ? sigma : NaN, n_train=n)

    return result, info, mach
end
# HACK: This is should properly figure out the inner model structure in the
# machine instead of relying on the fact that we know how the model is built
inner_model(machine::Machine, ::RegressionDataSet) = fitted_params(machine).transformed_target_model_deterministic.model.libsvm_model
inner_model(machine::Machine, ::CategoricalDataSet) = fitted_params(machine).svc.libsvm_model
inner_model(machine::Machine, ::MNIST) = fitted_params(machine).libsvm_model
inner_model(machine::Machine, ::DelveRegressionDataSet) = fitted_params(machine).transformed_target_model_deterministic.model.libsvm_model

# Fields in results that we don't want to collect in the final DataFrame
# since they are not relevant for the analysis and they take up a lot of space
default_ignore_results() = string.([:machine, :result, :per_fold])

# struct2dict from DrWatson but with string as the dictionary key
function struct2strdict(s)
    Dict(string(x) => getfield(s, x) for x in fieldnames(typeof(s)))
end

function produce_or_load(svm::SVMConfig; filename=default_savefile(svm), kwargs...)

    produce_or_load(svm; prefix="", suffix="", filename, tag=true, kwargs...) do ex
        perf, info, mach = run(ex)

        result = merge(struct2strdict(ex), struct2strdict(info))

        # From the performance evaluation, we save it into result and expose the
        # two most important fields: measurement and per_fold
        result["result"] = perf
        result["measurement"] = perf.measurement[1]
        result["per_fold"] = perf.per_fold[1]

        result["std"] = std(perf.per_fold[1])
        result["n_iter"] = sum(inner_model(mach, svm.dataset).n_iter)

        # We also save the fitted machine, so that we can use it later to make predictions
        # quickly
        result["machine"] = mach

        result
    end
end

"""
# Parameter grid for SVM

This returns a list of dictionaries with the parameters to use for creating SVMConfig
objects.
"""
function frenay_parameter_grid(; step::Float64=1.0, datasets=nothing)::Vector{Dict{Symbol,Any}}
    # Blacklist MNIST since it takes too long to run
    if datasets isa Nothing
        datasets = filter(DataSets.all) do d
            !(d in [DataSets.mnist])
        end
    end

    parameters_common = Dict(
        :dataset => datasets,
        :cost => [10 .^ (-2:step:3); @onlyif(:dataset isa DataSets.Small, 10 .^ ((3+step):step:6))],
        :epsilon => [
            @onlyif(is_regression(:dataset), 10 .^ (-5:step:1))
            @onlyif(!is_regression(:dataset), [0])
        ],
    )

    parameters_rbf = Dict(
        :kernel => LIBSVM.Kernel.RadialBasis,
        :gamma => 10 .^ (-3:step:0), parameters_common...
    )

    sigma_asin = 10 .^ (-3:step:3)

    parameters_asin = Dict(
        :kernel => [[LIBSVM.Kernel.Asin, LIBSVM.Kernel.AsinNorm]; @onlyif((:dataset isa DataSets.Small) && (:cost < 1e4), [LIBSVM.Kernel.Acos0, LIBSVM.Kernel.Acos1, LIBSVM.Kernel.Acos2])],
        :gamma => Utils.sigma2gamma.(sigma_asin),
        parameters_common...
    )

    [dict_list(parameters_asin); dict_list(parameters_rbf)]
end

function svm_parameter_grid(; step::Float64=1.0, datasets=nothing, acos=false, rbf=true, kwargs...)::Vector{Dict{Symbol,Any}}
    # Blacklist MNIST since it takes too long to run
    if datasets isa Nothing
        datasets = filter(DataSets.all) do d
            !(d in [DataSets.mnist])
        end
    end

    parameters_common = Dict(
        :dataset => datasets,
        :cost => 10 .^ (-2:step:4),
        :epsilon => [
            @onlyif(is_regression(:dataset), 10 .^ (-5:step:1))
            @onlyif(!is_regression(:dataset), [0])
        ],
        kwargs...
    )

    parameters_rbf =
        Dict(
            :kernel => LIBSVM.Kernel.RadialBasis,
            :gamma => 10 .^ (-3:step:0), parameters_common...
        )

    sigma_asin = 10 .^ (-3:step:6)

    parameters_asin = Dict(
        :kernel => [[LIBSVM.Kernel.Asin, LIBSVM.Kernel.AsinNorm]; @onlyif(acos, [LIBSVM.Kernel.Acos0, LIBSVM.Kernel.Acos1, LIBSVM.Kernel.Acos2])],
        :gamma => Utils.sigma2gamma.(sigma_asin),
        parameters_common...
    )

    if rbf
        return [dict_list(parameters_rbf); dict_list(parameters_asin)]
    end

    dict_list(parameters_asin)
end

function svm_parameter_grid_sigma_reg(; step::Float64=1.0, datasets=nothing, kwargs...)::Vector{Dict{Symbol,Any}}
    # Blacklist MNIST since it takes too long to run
    if datasets isa Nothing
        datasets = filter(DataSets.all) do d
            !(d in [DataSets.mnist])
        end
    end

    parameters = Dict(
        :dataset => datasets,
        :cost => 10 .^ (-2:step:4),
        :kernel => [[LIBSVM.Kernel.Asin, LIBSVM.Kernel.AsinNorm]; @onlyif(acos, [LIBSVM.Kernel.Acos0, LIBSVM.Kernel.Acos1, LIBSVM.Kernel.Acos2])],
        :sigma => 10 .^ (-3:step:3),
        :epsilon => 10 .^ (-5:step:1),
        kwargs...
    )

    dict_list(parameters)
end

end
