module Experiments

using Dates
using MLJ, MLJBase
using Base: @kwdef
using LIBSVM

import ..DataSets: DataSet
import ..Measures: MeanSquaredError
import ..Resampling
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
abstract type Experiment end

run!(ex::Experiment) = error("run!($(typeof(ex))) not implemented")
save(ex::Experiment, ::String) = error("save($(typeof(ex)), filename) not implemented")

abstract type MLJTunedModelExperimentResult end

struct SimpleMLJTunedModelExperimentResult
    # machine::Any
    # measure::Float64
    elapsed::Dates.Period
end


@kwdef mutable struct MLJTunedModelExperiment <: Experiment
    executed::Bool = false
    kernel_check::Bool = false

    date::Date = Date(Dates.now())
    host::String = gethostname()

    kernel::LIBSVM.Kernel.KERNEL
    dataset::DataSet
    basemodel::Union{Nothing, Model}

    resampling::ResamplingStrategy = CV(nfolds=5)
    partition::Union{Nothing, AbstractFloat} = nothing
    measure::MLJBase.Measure = MeanSquaredError()
    grid_step::AbstractFloat = 1.0

    start_time::Union{Nothing, Dates.TimeType} = nothing
    end_time::Union{Nothing, Dates.TimeType} = nothing

    result::Union{Nothing, SimpleMLJTunedModelExperimentResult} = nothing
end

function __ensure_kernel_match!(ex::MLJTunedModelExperiment; warn=true)
    if ex.kernel_check
        return
    end
    if ex.basemodel === nothing
        error("basemodel is not defined")
    end
    if ex.basemodel.kernel !== ex.kernel
        warn && @warn("basemodel.kernel changed from $(ex.basemodel.kernel) to $(ex.kernel)")
        ex.basemodel.kernel = ex.kernel
    end
    ex.kernel_check = true
end

function MLJTunedModelExperiment(kernel::LIBSVM.Kernel.KERNEL, dataset::DataSet, basemodel::Union{Nothing, Model}=nothing; model_params::Dict{Symbol, Any}=Dict{Symbol, Any}(), kwargs...)
    if basemodel === nothing
        basemodel = Models.basemodel(dataset)(;kernel, model_params...)
    end
    out = MLJTunedModelExperiment(;kernel, dataset, basemodel, kwargs...)
    return out
end

function run!(ex::MLJTunedModelExperiment; force=false)::SimpleMLJTunedModelExperimentResult
    if ex.executed && !force
        @warn("Experiment already executed, loading results. (use force=true to force re-execution))")
        return ex.result
    end
    __ensure_kernel_match!(ex)
    ex.executed = true
    ex.start_time = Dates.now()

    @info "Loading dataset $(ex.dataset)"
    X, y = if ex.partition === nothing
        @info "No partition defined, loading full dataset"
        unpack(ex.dataset)
    elseif 0.0 < ex.partition < 1.0
        @info "Partition defined, loading $(ex.partition) of dataset as training set"
        partition(ex.dataset, ex.partition)
    else
        error("Invalid partition value: $(ex.partition)")
    end
    @info "Loading dataset $(ex.dataset) done after $(Dates.now() - ex.start_time)"

    # Reset time, so that we measure only the time of the model fitting
    ex.start_time = Dates.now()

    ex.end_time = Dates.now()

    ex.result = SimpleMLJTunedModelExperimentResult(
        # r.best_model, r.measure,
        ex.end_time - ex.start_time)

    return ex.result
end

end # module Experiments
