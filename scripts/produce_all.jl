#!/usr/bin/env julia

using TFM
using LIBSVM
using Distributed
using DrWatson
using ProgressMeter

step = 1.0

datasets = filter(DataSets.all) do d
    # !(d in [DataSets.mnist, DataSets.cancer])
    d in [DataSets.ailerons, DataSets.triazines, DataSets.elevators] # 3 datasets that had issues
end

parameters_common = Dict(
    :dataset => datasets,
    # :cost => 10 .^ (-2:step:6),
    :cost => 10 .^ (-2:step:4),
    :epsilon => 10 .^ (-5:step:1),
)

parameters_rbf = Dict(
    :kernel => LIBSVM.Kernel.RadialBasis,
    :gamma => 10 .^ (-3:step:0), parameters_common...
)

sigma_asin = 10 .^ (-3:step:3)

parameters_asin = Dict(
    :kernel => [LIBSVM.Kernel.Asin, LIBSVM.Kernel.AsinNorm],
    :gamma => Utils.sigma2gamma.(sigma_asin),
    parameters_common...
)

parameters_all = [dict_list(parameters_asin) ; dict_list(parameters_rbf)]

@info "This will run $(length(parameters_all)) experiments ..."

@warn "Proceed? [y/n]"
readline() == "y" || error("Aborted.")

addprocs(10)

@everywhere begin
    using TFM.Experiments: SVMConfig, LoadedData
    using DrWatson: produce_or_load

    ld = LoadedData()
end

successes = @showprogress pmap(parameters_all) do params
    ex = SVMConfig(;params...)
    # WARNING: Remove the `force=true` argument once this special run is done
    _, file = produce_or_load(ex; loadfile=false, force=true)
    1
end

@info "Finished $(sum(successes)) experiments."
