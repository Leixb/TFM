#!/usr/bin/env julia

using TFM
using LIBSVM
using Distributed
using DrWatson

const parameters_all = TFM.Experiments.svm_parameter_grid()

@warn "Generated $(length(parameters_all)) executable combinations ..."

configs = map(parameters_all) do params
    TFM.SVMConfig(;params...)
end

configs = filter(configs) do c
    !isfile(Experiments.default_savefile(c))
end

@info "Skipping $(length(parameters_all) - length(configs)) executions which already exist..."
@info "Running remaining $(length(configs)) ..."

addprocs(10)

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
