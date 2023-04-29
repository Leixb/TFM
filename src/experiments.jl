module Experiments

using Dates
using MLJ, MLJBase
using Base: @kwdef
using LIBSVM

import ..DataSets: DataSet
import ..Measures: MeanSquaredError

"""
Experiment is the abstract type for all experiments.

It is used to store the metadata and results of an experiment to
be able to save it into a file and reproduce it later.

As such, all experiment types, must implement the following fields:
- date::Date: the date when the experiment was run.
- host::String: the hostname of the machine where the experiment was run.

And the following methods:
- run(experiment::Experiment): run the experiment and store the results.
- save(experiment::Experiment, filename::String): save the experiment to a file.
- load(filename::String)::Experiment: load an experiment from a file.
"""
abstract type Experiment end

run(ex::Experiment) = error("run($(typeof(ex))) not implemented")
save(ex::Experiment, ::String) = error("save($(typeof(ex)), filename) not implemented")

@kwdef struct MLJTunedModelExperiment <: Experiment
    date::Date = Date(Dates.now())
    host::String = gethostname()
    kernel::LIBSVM.Kernel.KERNEL
    basemodel::Model
    dataset::DataSet
    resampling::ResamplingStrategy = CV(nfolds=5)
    partition::Union{Nothing, AbstractFloat} = nothing
    measure::MLJBase.Measure = MeanSquaredError()
    grid_step::AbstractFloat = 1.0
end

struct MLJTunedModelExperimentResult
    machine::Any
    measure::Float64
end

@kwdef struct MLJExperiment <: Experiment
    model::Model
    dataset::DataSet
    partition::Tuple{Vector{Int}, Vector{Int}} = ([], [])
    julia::String = VERSION
end

function run(ex::MLJExperiment)
    X, y = ex.dataset()
    r = resample(ex.resampling, ex.model, X, y, measure=ex.measure)
    ex.results = r
    return ex
end

end # module Experiments
