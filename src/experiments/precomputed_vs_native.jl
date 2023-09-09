"""
# Experiment: Precomputed vs Native

Aim: Compare the performance of the precomputed kernel with the native kernel
implementation:

    Does the precomputed kernel perform significantly worse than implementing
    the kernel by ourselves?

## TODO list

- [ ] Implement the precomputed kernel with MLJ
- [ ] Run for all kernels
- [ ] Try increasing the number of features / samples
- [ ] Benchmark both time and memory
- [ ] Make sure that the results are the same in all cases

"""
module PrecomputedVsNativeExperiment

using LIBSVM, MLJ, MLJBase

import ..ExperimentParameters, ..default_resampling, ..default_measure, ..run, ..default_savefile
import ...DataSets: DataSet
import ...Models

@kwdef struct PrecomputedVsNativeConfig <: ExperimentParameters
    n_features::Int = 100

    dataset::DataSet

    resampling::ResamplingStrategy = default_resampling(dataset)
    measure::MLJBase.Measure = default_measure(dataset)

    kernel::LIBSVM.Kernel.KERNEL = LIBSVM.Kernel.RadialBasis
    cost::Float64 = 1.0
    gamma::Float64 = 0.0

    epsilon::Union{Float64,Nothing} = 0.1

    extra_params::Dict{Symbol,Any} = Dict{Symbol,Any}(:max_iter => Int32(1e5))
end

allaccess(::PrecomputedVsNativeConfig) = (
    :n_features, :dataset, :resampling, :measure, :kernel, :cost, :gamma,
    :epsilon,
)

function run(config::PrecomputedVsNativeConfig)
    start = Dates.now()

    (Xtrain, Xtest), (ytrain, ytest) = partition(config.dataset)

    model = Models.pipeline(config.dataset;)
    mach = machine(model, Xtrain, ytrain)

    result = evaluate!(mach, resampling=config.resampling, measure=config.measure)
    yhat = MLJ.predict(mach, Xtest)
    measure_test = config.measure(yhat, ytest)

    duration = Dates.now() - start
end

end
