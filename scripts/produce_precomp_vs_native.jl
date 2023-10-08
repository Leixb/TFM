#!/usr/bin/env julia

using TFM.Experiments
using TFM.DataSets
using LIBSVM
using Distributed
using DrWatson
using ProgressMeter

using Dates

folder = get(ARGS, 1, "precomputed_vs_native")

PROCS = get(ENV, "PROCS", max(1, Sys.CPU_THREADS - 1))
if PROCS isa String
    PROCS = parse(Int, PROCS)
end

REPEATS = get(ENV, "REPEATS", 5)
if REPEATS isa String
    REPEATS = parse(Int, REPEATS)
end

@info "Using $PROCS processes"

start = Dates.now()
@info "Generating parameters ..."

const parameters_all = Experiments.PrecomputedVsNativeExperiment.parameter_grid()

@info "dict_list Done in $(Dates.now() - start)"

@warn "Generated $(length(parameters_all)) executable combinations ..."

start = Dates.now()

configs = @showprogress map(parameters_all) do params
    Experiments.PrecomputedVsNativeExperiment.PrecomputedVsNativeConfig(; params...)
end

@info "to SVM object Done in $(Dates.now() - start)"

if length(configs) == 0
    @info "Nothing to do."
    exit(0)
end

addprocs(PROCS)

@everywhere begin
    using DrWatson: produce_or_load
    using ProgressMeter
    using TFM
end

successes = @showprogress pmap(configs) do ex
    produce_or_load(ex; loadfile=false)
    1
end

@info "Finished $(sum(successes)) experiments."
