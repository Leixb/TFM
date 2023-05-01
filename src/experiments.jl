module Experiments

using Dates
using MLJ, MLJBase
using LIBSVM

import DrWatson: allaccess, default_prefix, default_allowed

import ..DataSets: DataSet, CategoricalDataSet, RegressionDataSet
import ..Measures: MeanSquaredError
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

- `run!(experiment::Experiment)`: run the experiment and store the results.
- `save(experiment::Experiment, filename::String)`: save the experiment to a file.
- `load(filename::String)::Experiment`: load an experiment from a file.

## References

- [DrWatson: Saving Tools](https://juliadynamics.github.io/DrWatson.jl/dev/save/)

"""
Base.@kwdef struct ExecutionInfo <: TFMType
    date::Date = Date(Dates.now())
    host::String = gethostname()
    duration::Dates.Period
end

# SVM without using tuning from MLJ

abstract type SVMConfig <: TFMType end

Base.@kwdef mutable struct EpsilonSVRConfig <: SVMConfig
    dataset::RegressionDataSet

    # Evaluation parameters
    resampling::ResamplingStrategy = CV(nfolds=5)
    measure::MLJBase.Measure = MeanSquaredError()

    # Model parameters
    kernel::LIBSVM.Kernel.KERNEL = LIBSVM.Kernel.RadialBasis
    cost::Float64 = 1.0
    gamma::Float64 = 0.0
    epsilon::Float64 = 0.1
    extra_params::Dict{Symbol, Any} = Dict{Symbol, Any}()

    # Execution metadata
    info::Union{Nothing, ExecutionInfo} = nothing
    mach::Union{Nothing, MLJBase.Machine} = nothing
    result::Union{Nothing, PerformanceEvaluation} = nothing
end

Base.@kwdef mutable struct SVCConfig <: SVMConfig
    dataset::CategoricalDataSet

    # Evaluation parameters
    resampling::ResamplingStrategy = CV(nfolds=5)
    measure::MLJBase.Measure = MeanSquaredError()

    # Model parameters
    kernel::LIBSVM.Kernel.KERNEL = LIBSVM.Kernel.RadialBasis
    cost::Float64 = 1.0
    gamma::Float64 = 0.0
    extra_params::Dict{Symbol, Any} = Dict{Symbol, Any}()

    # Execution metadata
    info::Union{Nothing, ExecutionInfo} = nothing
    mach::Union{Nothing, MLJBase.Machine} = nothing
    result::Union{Nothing, PerformanceEvaluation} = nothing
end

model_parameters(::EpsilonSVRConfig) = [ :kernel, :cost, :gamma, :epsilon ]
model_parameters(::SVCConfig) = [ :kernel, :cost, :gamma ]

# savename configuration

allaccess(svm::SVMConfig) = [ :dataset, :resampling, :measure, model_parameters(svm)...]

default_prefix(ex::EpsilonSVRConfig) = "svr_$(ex.info.date)"
default_prefix(ex::SVCConfig) = "svc_$(ex.info.date)"

default_allowed(::TFMType) = (
    Real, String, Symbol, TimeType, Kernel.KERNEL, DataSet, ResamplingStrategy, MLJBase.Measure
)

function model(svm::SVMConfig)
    parameters = map(model_parameters(svm)) do param
        (param, getfield(svm, param))
    end

    Models.pipeline(svm.dataset; parameters..., svm.extra_params...)
end

function run!(svm::SVMConfig)::PerformanceEvaluation
    start = Dates.now()

    X, y = unpack(svm.dataset)
    svm.mach = machine(model(svm), X, y)

    svm.result = evaluate!(svm.mach; svm.resampling, svm.measure)

    duration = Dates.now() - start
    svm.info = ExecutionInfo(;duration)

    return svm.result
end

end # module Experiments
