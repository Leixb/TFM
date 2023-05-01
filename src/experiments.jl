"""
# Experiment

This module defines how experiments are run and saved.

Under the hood, it uses [DrWatson](https://juliadynamics.github.io/DrWatson.jl/dev/), which enables
us to save the results of experiments in a structured way and
to easily load them back into Julia for further analysis.

## produce_or_load

Thanks to DrWatson, we can use the `produce_or_load` macro to run an experiment
and save the results to disk. If the experiment has already been run, it will
load the results from disk instead.

## Workflow

The typical workflow is as follows:

```julia
ex = EpsilonSVRConfig(DataSets.cpu, kernel=LIBSVM.Kernel.RadialBasis, cost=1.0, gamma=0.0, epsilon=0.1)
results, file = produce_or_load(ex)
```

Using `dict_list` we can, generate all the possible combinations of parameters
for a given experiment:

```julia
c = Dict(
    :cost => [0.1, 1.0, 10.0],
    :gamma => [0.0, 0.1, 1.0],
    :epsilon => [0.1, 0.2, 0.3]
)
grid = map(dict_list(c)) do params
    EpsilonSVRConfig(;dataset=DataSets.cpu, params...)
end

results = map(produce_or_load, grid)
```

Finally, we can make use of the `collect_results` function to collect the
results of multiple experiments into a single DataFrame:

```julia
df = collect_results!(datadir())
```

## References

- [DrWatson: Saving Tools](https://juliadynamics.github.io/DrWatson.jl/dev/save/)

"""
module Experiments

using Dates
using MLJ, MLJBase
using LIBSVM

using DrWatson
import DrWatson: allaccess, default_prefix, default_allowed, produce_or_load

import ..DataSets: DataSet, CategoricalDataSet, RegressionDataSet
import ..Measures: MeanSquaredError
import ..Models

import ..TFMType

Base.@kwdef struct ExecutionInfo <: TFMType
    date::Date = Date(Dates.now())
    host::String = gethostname()
    duration::Dates.Period
end

# SVM without using tuning from MLJ

abstract type SVMConfig <: TFMType end

Base.@kwdef struct EpsilonSVRConfig <: SVMConfig
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
end

Base.@kwdef struct SVCConfig <: SVMConfig
    dataset::CategoricalDataSet

    # Evaluation parameters
    resampling::ResamplingStrategy = CV(nfolds=5)
    measure::MLJBase.Measure = MeanSquaredError()

    # Model parameters
    kernel::LIBSVM.Kernel.KERNEL = LIBSVM.Kernel.RadialBasis
    cost::Float64 = 1.0
    gamma::Float64 = 0.0
    extra_params::Dict{Symbol, Any} = Dict{Symbol, Any}()
end

model_parameters(::EpsilonSVRConfig) = [ :kernel, :cost, :gamma, :epsilon ]
model_parameters(::SVCConfig) = [ :kernel, :cost, :gamma ]

# savename configuration

allaccess(svm::SVMConfig) = [ :dataset, :resampling, :measure, model_parameters(svm)...]

default_prefix(::SVMConfig) = "SVM"
default_prefix(::EpsilonSVRConfig) = "SVR"
default_prefix(::SVCConfig) = "SVC"

default_allowed(::TFMType) = (
    Real, String, Symbol, TimeType, Kernel.KERNEL, DataSet, ResamplingStrategy, MLJBase.Measure
)

function model(svm::SVMConfig)
    parameters = map(model_parameters(svm)) do param
        (param, getfield(svm, param))
    end

    Models.pipeline(svm.dataset; parameters..., svm.extra_params...)
end

Base.@kwdef struct LoadedData <: TFMType
    data::Dict{DataSet, Tuple{Any, Any}} = Dict{DataSet, Tuple{Any, Any}}()
end

function cached_unpack!(data::LoadedData, ds::DataSet)::Tuple{Any, Any}
    if haskey(data.data, ds)
        return data.data[ds]
    else
        X, y = unpack(ds)
        data.data[ds] = (X, y)
        return X, y
    end
end

function run(svm::SVMConfig, data::LoadedData)::Tuple{PerformanceEvaluation, ExecutionInfo, Machine}
    X, y = cached_unpack!(data, svm.dataset)
    run(svm, X, y)
end

function run(svm::SVMConfig)::Tuple{PerformanceEvaluation, ExecutionInfo, Machine}
    X, y = unpack(svm.dataset)
    run(svm, X, y)
end

function run(svm::SVMConfig, X, y)::Tuple{PerformanceEvaluation, ExecutionInfo, Machine}
    start = Dates.now()

    mach = machine(model(svm), X, y)

    result = evaluate!(mach; svm.resampling, svm.measure)

    duration = Dates.now() - start
    info = ExecutionInfo(;duration)

    return result, info, mach
end

default_savefile(svm::SVMConfig) = datadir("svms", savename(svm, "jld2"))

function load(svm::SVMConfig; filename::String=default_savefile(svm))
    @info "Loading $(filename)"
    wload(filename)
end

function produce_or_load(svm::SVMConfig; cache::Union{Nothing, LoadedData}=nothing, filename=default_savefile(svm), kwargs...)

    method = if cache === nothing
        run
    else
        x -> run(x, cache)
    end

    produce_or_load(svm; prefix="", suffix="", filename, tag=true, kwargs...) do ex
        perf, info, mach = method(ex)

        result = merge(struct2dict(ex), struct2dict(info))

        # From the performance evaluation, we save it into result and expose the
        # two most important fields: measurement and per_fold
        result[:result] = perf
        result[:measurement] = perf.measurement[1]
        result[:per_fold] = perf.per_fold[1]

        # We also save the fitted machine, so that we can use it later to make predictions
        # quickly
        result[:machine] = mach

        result
    end
end

end # module Experiments
