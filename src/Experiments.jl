module Experiments

using Dates
using MLJBase
using MLJ
using Base: @kwdef

import ..DataSets: DataSet

@kwdef struct Experiment
    date::Date = Date(Dates.now())
    host::String = gethostname()
end

@kwdef struct MLJExperiment <: Experiment
    model::MLJ.MLJBase.MLJType
    dataset::DataSet
    resampling::ResamplingStrategy
    measure::MLJBase.Measure
    seed::Int = 1234
end

end # module Experiments
