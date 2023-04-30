module Experiments

using Dates
using MLJ, MLJBase
using Base: @kwdef
using LIBSVM

import DrWatson: allaccess, default_prefix, default_allowed

import ..DataSets: DataSet, CategoricalDataSet, RegressionDataSet
import ..Measures: MeanSquaredError
import ..Resampling: TwoFold, FiveTwo, RepeatedCV
import ..Models

import ..TFMType

"""
# Experiment

Abstract type for all experiments.

It is used to store the **metadata**, **parameters** and **results** of an experiment to
be able to save it into a file and *to store and reproduce it* later.

## Fields

- `date::Date`: the date when the experiment was run.
- `host::String`: the *hostname* of the machine where the experiment was run.

## Interface

- `run(experiment::Experiment)`: run the experiment and store the results.
- `save(experiment::Experiment, filename::String)`: save the experiment to a file.
- `load(filename::String)::Experiment`: load an experiment from a file.

## References

- [DrWatson: Saving Tools](https://juliadynamics.github.io/DrWatson.jl/dev/save/)

"""
abstract type Experiment <: TFMType end

default_prefix(::Experiment) = "experiment"
default_allowed(::Experiment) = (
    Real, String, Symbol, TimeType, Kernel.KERNEL, DataSet
)

default_allowed

run!(ex::Experiment) = error("run!($(typeof(ex))) not implemented")
save(ex::Experiment, ::String) = error("save($(typeof(ex)), filename) not implemented")

abstract type MLJTunedModelExperimentResult end

struct SimpleMLJTunedModelExperimentResult
    machine::Any
    # measure::Float64
    elapsed::Dates.Period
end


@kwdef mutable struct MLJTunedModelExperiment <: Experiment
    executed::Bool = false

    date::Date = Date(Dates.now())
    host::String = gethostname()

    kernel::LIBSVM.Kernel.KERNEL
    dataset::DataSet
    model::Union{Nothing, Model}

    resampling::ResamplingStrategy = CV(nfolds=5)
    partition::Union{Nothing, AbstractFloat} = nothing
    measure::MLJBase.Measure = MeanSquaredError()
    grid_step::AbstractFloat = 1.0

    start_time::Union{Nothing, Dates.TimeType} = nothing
    end_time::Union{Nothing, Dates.TimeType} = nothing

    result::Union{Nothing, SimpleMLJTunedModelExperimentResult} = nothing
end

allaccess(::MLJTunedModelExperiment) = [
    :kernel, :dataset, :grid_step, :measure
]
default_prefix(ex::MLJTunedModelExperiment) = "experiment_mlj_$(ex.date)"

function MLJTunedModelExperiment(kernel::LIBSVM.Kernel.KERNEL, dataset::DataSet, model::Union{Nothing, Model}=nothing; model_params::Dict{Symbol, Any}=Dict{Symbol, Any}(), kwargs...)
    if model === nothing
        model = Models.pipeline(dataset; kernel, model_params...)
    end
    out = MLJTunedModelExperiment(;kernel, dataset, model, kwargs...)
    return out
end

function run!(ex::MLJTunedModelExperiment; force=false)::SimpleMLJTunedModelExperimentResult
    if ex.executed && !force
        @warn("Experiment already executed, loading results. (use force=true to force re-execution))")
        return ex.result
    end
    ex.executed = true
    ex.start_time = Dates.now()

    @info "Loading dataset $(ex.dataset)"
        Xtrain, ytrain = if ex.partition === nothing
        @info "No partition defined, loading full dataset"
        unpack(ex.dataset)
    elseif 0.0 < ex.partition < 1.0
        @info "Partition defined, loading $(ex.partition) of dataset as training set"
        (Xtrain, Xtest), (ytrain, ytest) = partition(ex.dataset, ex.partition)
    else
        error("Invalid partition value: $(ex.partition)")
    end
    @info "Loading dataset $(ex.dataset) done after $(Dates.now() - ex.start_time)"

    # Reset time, so that we measure only the time of the model fitting
    ex.start_time = Dates.now()

    mach = machine(ex.model, Xtrain, ytrain)
    MLJ.fit!(mach)

    ex.end_time = Dates.now()

    ex.result = SimpleMLJTunedModelExperimentResult(
        mach,
        # r.best_model, r.measure,
        ex.end_time - ex.start_time)

    return ex.result
end

"""
Execution metadata
"""
@kwdef struct ExecutionInfo <: TFMType
    date::Date = Date(Dates.now())
    host::String = gethostname()

    start_time::Union{Nothing, Dates.TimeType} = nothing
    end_time::Union{Nothing, Dates.TimeType} = nothing
    duration::Union{Nothing, Dates.Period} = nothing
end

# SVM without using tuning from MLJ

abstract type SVMConfig <: TFMType end

@kwdef struct EpsilonSVRConfig <: SVMConfig
    dataset::RegressionDataSet

    # Evaluation parameters
    resampling::ResamplingStrategy = CV(nfolds=5)
    measure::MLJBase.Measure = RootMeanSquaredError()

    # Model parameters
    kernel::LIBSVM.Kernel.KERNEL = LIBSVM.Kernel.RadialBasis
    cost::Float64 = 1.0
    gamma::Float64 = 0.0
    epsilon::Float64 = 0.1
    extra_params::Dict{Symbol, Any} = Dict{Symbol, Any}()

    # Execution metadata
    info::ExecutionInfo = ExecutionInfo()

    result::Union{Nothing, PerformanceEvaluation} = nothing
end

@kwdef struct SVCConfig <: SVMConfig
    dataset::CategoricalDataSet

    # Evaluation parameters
    resampling::ResamplingStrategy = CV(nfolds=5)
    measure::MLJBase.Measure = RootMeanSquaredError()

    kernel::LIBSVM.Kernel.KERNEL = LIBSVM.Kernel.RadialBasis
    cost::Float64 = 1.0
    gamma::Float64 = 0.0
    extra_params::Dict{Symbol, Any} = Dict{Symbol, Any}()

    # Execution metadata
    info::Base.RefValue{ExecutionInfo} = C_NULL
    result::Base.RefValue{PerformanceEvaluation} = C_NULL
end

# savename configuration

allaccess(::EpsilonSVRConfig) = [ :dataset, :resampling, :kernel, :cost, :gamma, :epsilon ]
allaccess(::SVCConfig) = [ :dataset, :resampling, :kernel, :cost, :gamma ]

default_prefix(ex::EpsilonSVRConfig) = "svr_$(ex.info.date)"
default_prefix(ex::SVCConfig) = "svc_$(ex.info.date)"

default_allowed(::TFMType) = (
    Real, String, Symbol, TimeType, Kernel.KERNEL, DataSet, ResamplingStrategy
)

model(ex::SVMConfig) = error("model($(typeof(ex))) not implemented")

model(ex::EpsilonSVRConfig) = LIBSVM.EpsilonSVR(
    kernel=ex.kernel,
    cost=ex.cost,
    gamma=ex.gamma,
    epsilon=ex.epsilon,
    ex.extra_params...
)

model(ex::SVCConfig) = LIBSVM.SVC(
    kernel=ex.kernel,
    cost=ex.cost,
    gamma=ex.gamma,
    ex.extra_params...
)

function run(svm::SVMConfig)::PerformanceEvaluation
    model = Models.pipeline(svm.dataset; svm.kernel, svm.cost, svm.gamma, svm.extra_params...)

    # svm.info[] = ExecutionInfo()
    # svm.info[].start_time = Dates.now()

    X, y = unpack(svm.dataset)
    mach = machine(model, X, y)

    result = evaluate!(mach; resampling=svm.resampling, measure=svm.measure)

    # svm.info[].end_time = Dates.now()
    # svm.info[].duration = svm.info[].end_time - svm.info[].start_time

    # svm.result[] = result

    return result
end

end # module Experiments
