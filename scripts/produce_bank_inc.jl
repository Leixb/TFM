#!/usr/bin/env julia

using TFM.Experiments
using TFM.DataSets
using LIBSVM
using Distributed
using DrWatson

folder = get(ARGS, 1, "svms4_inc")

datasets = [DataSets.Bank32fm()]

const parameters_all = Experiments.svm_parameter_grid(;datasets, acos=false, rbf=false, step=1.0, folder, scale_sigma=true,
    subsample=[0.2:0.2:0.8; nothing]
)

@warn "Generated $(length(parameters_all)) executable combinations ..."

configs = map(parameters_all) do params
    Experiments.SVMConfig(;params...)
end

data_path = datadir(folder)
if !isdir(data_path)
    mkpath(data_path)
else
    files_done = Set(readdir(data_path))
    configs = filter(configs) do c
        !(savename(c, "jld2") in files_done)
    end
end

@info "Skipping $(length(parameters_all) - length(configs)) executions which already exist..."
@info "Running remaining $(length(configs)) ..."

addprocs(11)

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
