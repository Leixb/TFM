"""
# Experiment: Precomputed vs Native

Aim: Compare the performance of the precomputed kernel with the native kernel
implementation:

    Does the precomputed kernel perform significantly worse than implementing
    the kernel by ourselves?

## TODO list

- [x] Implement the precomputed kernel with MLJ
- [ ] Run for all kernels
- [ ] Try increasing the number of features / samples
- [ ] Benchmark both time and memory
- [x] Make sure that the results are the same in all cases

"""
module PrecomputedVsNativeExperiment

using LIBSVM, MLJ, MLJBase, Dates, DrWatson

import DrWatson: allaccess, default_prefix, default_allowed, produce_or_load

import ..ExperimentParameters, ..default_resampling, ..default_measure, ..run, ..default_savefile
import ...DataSets: DataSet
import ...Models, ...Utils, ...DataSets

function kernel_pairings()
    Dict([
        (LIBSVM.Kernel.Asin, Utils.kernel_asin),
        (LIBSVM.Kernel.AsinNorm, Utils.kernel_asin_normalized),
        # (LIBSVM.Kernel.Acos0, Utils.kernel_acos_0),
        # # (LIBSVM.Kernel.Acos1, Utils.kernel_acos_1),
        # (LIBSVM.Kernel.Acos2, Utils.kernel_acos_2),
        # (LIBSVM.Kernel.Acos1Norm, Utils.kernel_acos_1_norm),
        # (LIBSVM.Kernel.Acos2Norm, Utils.kernel_acos_2_norm)
    ])
end

function parameter_grid(datasets=nothing)
    if datasets isa Nothing
        datasets = filter(DataSets.all) do d
            !(d in [DataSets.mnist])
        end
    end

    dict_list(Dict(
        :dataset => datasets,
        :kernel => collect(keys(kernel_pairings()))
    ))
end

@kwdef struct PrecomputedVsNativeConfig <: ExperimentParameters
    dataset::DataSet

    resampling::ResamplingStrategy = default_resampling(dataset)
    measure::MLJBase.Measure = default_measure(dataset)

    kernel::LIBSVM.Kernel.KERNEL = LIBSVM.Kernel.RadialBasis
    repeat::Int = 5

    extra_params::Dict{Symbol,Any} = Dict{Symbol,Any}(:max_iter => Int32(1e5))
end

default_prefix(svm::PrecomputedVsNativeConfig) = "preVsNative"
default_savefile(svm::PrecomputedVsNativeConfig) = datadir("precomputed_vs_native", savename(svm, "jld2"))

allaccess(::PrecomputedVsNativeConfig) = (
    :dataset, :resampling, :measure, :kernel, :repeat
)

function run(config::PrecomputedVsNativeConfig)

    (Xtrain, Xtest), (ytrain, ytest) = partition(config.dataset)

    kernel_precomputed = kernel_pairings()[config.kernel]

    model_native = Models.pipeline(config.dataset; config.kernel, gamma=0.5, config.extra_params...)
    model_precomputed = Models.pipeline(config.dataset; kernel=kernel_precomputed, config.extra_params...)

    # First, we run the models to precompile them (Julia's JIT compiler would otherwise make the first run much slower)

    # Clone the resampling strategy, so that we can run the model twice with the same seed
    resampler_clone = copy(config.resampling)

    run_one(model_native, Xtrain, ytrain, Xtest, ytest, config.resampling, config.measure)
    run_one(model_precomputed, Xtrain, ytrain, Xtest, ytest, resampler_clone, config.measure)

    times_native = []
    times_precomputed = []

    bytes_native = []
    bytes_precomputed = []

    gc_native = []
    gc_precomputed = []

    for i in 1:config.repeat
        native = run_one(model_native, Xtrain, ytrain, Xtest, ytest, config.resampling, config.measure)
        precomputed = run_one(model_precomputed, Xtrain, ytrain, Xtest, ytest, resampler_clone, config.measure)

        times_native = [times_native; native.time]
        times_precomputed = [times_precomputed; precomputed.time]

        bytes_native = [bytes_native; native.bytes]
        bytes_precomputed = [bytes_precomputed; precomputed.bytes]

        gc_native = [gc_native; native.gctime]
        gc_precomputed = [gc_precomputed; precomputed.gctime]
    end

    mean_time_native = mean(times_native)
    mean_time_precomputed = mean(times_precomputed)
    std_time_native = std(times_native)
    std_time_precomputed = std(times_precomputed)

    speedup = mean_time_precomputed / mean_time_native
    speedup_std = speedup * sqrt((std_time_native / mean_time_native)^2 + (std_time_precomputed / mean_time_precomputed)^2)

    return Dict(
        :dataset => config.dataset,
        :kernel => config.kernel,
        :resampling => config.resampling,
        :measure => config.measure,
        :times_native => times_native,
        :times_precomputed => times_precomputed,
        :mean_time_native => mean_time_native,
        :mean_time_precomputed => mean_time_precomputed,
        :std_time_native => std_time_native,
        :std_time_precomputed => std_time_precomputed,
        :speedup => speedup,
        :speedup_std => speedup_std,
        :bytes_native => bytes_native,
        :bytes_precomputed => bytes_precomputed,
        :gc_native => gc_native,
        :gc_precomputed => gc_precomputed,
        :mean_bytes_native => mean(bytes_native),
        :mean_bytes_precomputed => mean(bytes_precomputed),
    )
end

function run_one(model, Xtrain, ytrain, Xtest, ytest, resampling, measure)
    mach = machine(model, Xtrain, ytrain)

    timed = @timed evaluate!(mach, resampling=resampling, measure=measure)
    yhat = MLJ.predict(mach, Xtest)
    measure_test = measure(yhat, ytest)

    result = timed.value

    return (; result, measure_test, timed.time, timed.bytes, timed.gctime)
end

function produce_or_load(config::PrecomputedVsNativeConfig; loadfile=default_savefile(config))
    produce_or_load(config, run, loadfile=loadfile)
end

end
