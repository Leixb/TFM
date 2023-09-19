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
ex = SVMConfig(DataSets.cpu, kernel=LIBSVM.Kernel.RadialBasis, cost=1.0, gamma=0.0, epsilon=0.1)
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
    SVMConfig(;dataset=DataSets.cpu, params...)
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

using DataFrames, DataFramesMeta, DrWatson, Printf
import DrWatson: allaccess, default_prefix, default_allowed, produce_or_load

import ..DataSets: DataSet, CategoricalDataSet, RegressionDataSet, MNIST, DelveRegressionDataSet
import ..DataSets
import ..Measures: MeanSquaredError, NormalizedRootMeanSquaredError
import ..Models
import ..Utils
import ..Resampling

import ..DataSets: is_regression

import ..TFMType

"""
    ExecutionInfo <: TFMType

This type is used to store information about the execution of an experiment.
The fields will be saved to disk along with the results of the execution itself.

## Fields

  - `date`: The date when the experiment was run
  - `host`: The hostname of the machine where the experiment was run
  - `measure_test`: The value of the measure on the test set
  - `duration`: The duration of the experiment
  - `n_train`: The number of training samples
  - `sigma_scaled`: The value of sigma used for the RBF kernel, scaled by the
    number of training samples
"""
@kwdef struct ExecutionInfo <: TFMType
    date::Date = Date(Dates.now())
    host::String = gethostname()
    measure_test::Float64 = NaN
    duration::Dates.Period
    n_train::Int = -1
    sigma_scaled::Float64 = NaN
end


"""
    default_resampling(::DataSet)

Set the default resampling method to use for the datasets
"""
default_resampling(::DataSet) = Resampling.FiveTwo(1234)

"""
    default_measure(::DataSet)

The default measure to use for the datasets, if a dataset requires a specific
re-sampling method, it should override this function.
"""
default_measure(::Union{RegressionDataSet,DelveRegressionDataSet}) = NormalizedRootMeanSquaredError()
default_measure(::CategoricalDataSet) = Accuracy()


"""
    ExperimentParameters <: TFMType

This type is used to store the configuration of an experiment. The fields will
be saved to disk along with the results of the execution itself and they
determine all parameters of an execution.
"""
abstract type ExperimentParameters <: TFMType end

allaccess(::ExperimentParameters) = error("Not implemented")
run(::ExperimentParameters) = error("Not implemented")
default_savefile(::ExperimentParameters) = error("Not implemented")

default_prefix(::ExperimentParameters) = "EX"
default_allowed(::ExperimentParameters) = (
    Real, String, Symbol, TimeType, Kernel.KERNEL, DataSet, ResamplingStrategy,
    MLJBase.Measure
)

function load(ex::ExperimentParameters; filename::String=default_savefile(ex))
    @info "Loading $(filename)"
    wload(filename)
end

export experiment_data

function experiment_data(folder="svms", scan=true; kwargs...)
    if scan
        df = collect_results!(
            datadir(folder);
            black_list=Experiments.SVMExperiment.default_ignore_results(),
            kwargs...
        )
    else
        df = wload(datadir("results_$folder.jld2"))["df"]
    end
    df.kernel_cat = categorical(string.(df.kernel))
    df.dataset_cat = categorical(string.(df.dataset))
    df.sigma = Utils.gamma2sigma.(df.gamma)
    df.kernel_family = map(x -> string(x)[1:4], df.kernel_cat)
    df.cost = round.(df.cost, sigdigits=2)
    df.cost_cat = map(df.cost) do cost
        @sprintf("%.0E", cost)
    end
    df.measure_cv = df.measurement
    df.ms = @. Dates.value(df.duration)
    df.ms_per_iter = @. df.ms / df.n_iter / 5
    @rsubset(df, !(:dataset isa DataSets.Servo))
end

# Specific experiment configurations

include("./experiments/svm.jl")
using .SVMExperiment

include("./experiments/precomputed_vs_native.jl")
using .PrecomputedVsNativeExperiment

end # module Experiments
