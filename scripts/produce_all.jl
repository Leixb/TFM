#!/usr/bin/env julia

using TFM.Experiments
using LIBSVM
using Distributed
using DrWatson

const parameters_all = Experiments.svm_parameter_grid(;acos=false, step=2.0, folder="svms_2")

@warn "Generated $(length(parameters_all)) executable combinations ..."

configs = map(parameters_all) do params
    Experiments.SVMConfig(;params...)
end

# files_done = Set(readdir(datadir(default_params.folder)))
# configs = filter(configs) do c
#     !(savename(c, "jld2") in files_done)
# end

# @info "Skipping $(length(parameters_all) - length(configs)) executions which already exist..."
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
