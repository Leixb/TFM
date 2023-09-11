#!/usr/bin/env julia

using TFM.Experiments
using TFM.DataSets
using LIBSVM
using Distributed
using DrWatson
using ProgressMeter

using Dates

folder = get(ARGS, 1, "svms3")

PROCS = parse(Int, get(ENV, "PROCS", max(1, nprocs() - 1)))

@info "Using $PROCS processes"

# WARN: MNIST is too slow, if needed, run it separately
datasets = DataSets.all |> filter(ds -> !(ds isa DataSets.MNIST))

start = Dates.now()
@info "Generating parameters ..."

const parameters_all = Experiments.svm_parameter_grid(; datasets, acos=false, rbf=true, step=1.0, folder, scale_sigma=true)

@info "dict_list Done in $(Dates.now() - start)"

@warn "Generated $(length(parameters_all)) executable combinations ..."

start = Dates.now()

configs = @showprogress map(parameters_all) do params
    Experiments.SVMConfig(; params...)
end

@info "to SVM object Done in $(Dates.now() - start)"

start = Dates.now()

data_path = datadir(folder)
if !isdir(data_path)
    mkpath(data_path)
else
    files_done = Set(readdir(data_path))
    configs = filter(configs) do c
        !(savename(c, "jld2") in files_done)
    end
end

@info "Filter done in $(Dates.now() - start)"

@info "Skipping $(length(parameters_all) - length(configs)) executions which already exist..."
@info "Running remaining $(length(configs)) ..."

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
