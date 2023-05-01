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

has_run(ex::SVMConfig) = ex.info isa ExecutionInfo
_assert_has_run(ex::SVMConfig) = @assert has_run(ex) "$(ex) has not been run yet."

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

function run!(svm::SVMConfig)::PerformanceEvaluation
    start = Dates.now()

    X, y = unpack(svm.dataset)
    svm.mach = machine(model(svm), X, y)

    svm.result = evaluate!(svm.mach; svm.resampling, svm.measure)

    duration = Dates.now() - start
    svm.info = ExecutionInfo(;duration)

    return svm.result
end

default_savefile(svm::SVMConfig) = datadir(savename(svm, "jld2"))

function save(svm::SVMConfig; filename::String=default_savefile(svm))
    _assert_has_run(svm)
    @info "Saving to $(filename)"
    tagsave(filename, struct2dict(svm))
end

function load(svm::SVMConfig; filename::String=default_savefile(svm))
    @info "Loading $(filename)"
    wload(filename)
end

produce_or_load(svm::SVMConfig) = produce_or_load(svm; filename=default_savefile(svm)) do ex
    run!(ex)
    struct2dict(ex)
end

end # module Experiments
