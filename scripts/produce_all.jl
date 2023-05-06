#!/usr/bin/env julia

using TFM
using LIBSVM
using Distributed
using DrWatson
using ProgressMeter

step = 1.0

# Blacklist MNIST since it takes too long to run
datasets = filter(DataSets.all) do d
    !(d in [DataSets.mnist])
end

parameters_common = Dict(
    :dataset => datasets,
    :cost => [ 10 .^ (-2:step:3) ; @onlyif(:dataset isa DataSets.Small, 10 .^ ((3+step):step:6)) ],
    :epsilon => 10 .^ (-5:step:1),
)

parameters_rbf = Dict(
    :kernel => LIBSVM.Kernel.RadialBasis,
    :gamma => 10 .^ (-3:step:0), parameters_common...
)

sigma_asin = 10 .^ (-3:step:3)

parameters_asin = Dict(
    :kernel => [LIBSVM.Kernel.Asin, LIBSVM.Kernel.AsinNorm, LIBSVM.Kernel.Acos0, LIBSVM.Kernel.Acos1, LIBSVM.Kernel.Acos2],
    :gamma => Utils.sigma2gamma.(sigma_asin),
    parameters_common...
)

parameters_all = [dict_list(parameters_asin) ; dict_list(parameters_rbf)]

@warn "This will run $(length(parameters_all)) experiments ..."

using TFM.Experiments: SVMConfig
configs = map(parameters_all) do params SVMConfig(;params...) end

configs = filter(configs) do c
    !isfile(Experiments.default_savefile(c))
end

@info "Skipping $(length(parameters_all) - length(configs)) experiments already done..."
@info "Running $(length(configs)) experiments ..."

addprocs(10)

@everywhere begin
    using DrWatson: produce_or_load
end

successes = @showprogress pmap(configs) do ex
    _, file = produce_or_load(ex; loadfile=false)
    1
end

@info "Finished $(sum(successes)) experiments."
