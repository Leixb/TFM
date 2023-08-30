#!/usr/bin/env julia

using TFM
using LIBSVM
using Distributed
using DrWatson
using Random

datasets::Vector{DataSets.DataSet} = map(rand(MersenneTwister(1234), UInt32, (5))) do rng
    DataSets.Blobs(; rng)
end

foreach(print, datasets)

const parameters_all = Experiments.svm_parameter_grid(; datasets)

@warn "Generated $(length(parameters_all)) executable combinations ..."

configs = map(parameters_all) do params
    Experiments.SVMConfig(; params...)
end

files_done = Set(readdir(datadir("svms")))
configs = filter(configs) do c
    !(savename(c, "jld2") in files_done)
end

@info "Skipping $(length(parameters_all) - length(configs)) executions which already exist..."
@info "Running remaining $(length(configs)) ..."

if length(configs) == 0
    @info "Nothing to do."
    exit(0)
end

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
