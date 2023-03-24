#!/usr/bin/env julia

using DataFrames: DataFrame
using MLJ
using LIBSVM.Kernel

function prepare_dataset(df::DataFrame, target::Symbol, ratio=0.9, rng=1234)::Tuple{DataFrame,DataFrame,AbstractVector{Number},AbstractVector{Number}}
    y, X = unpack(df, ==(target); shuffle=false)

    (Xtrain, Xtest), (ytrain, ytest) = partition((X, y), ratio, rng=rng, multi=true)

    return Xtrain, Xtest, ytrain, ytest
end

function build_machine(model=@load EpsilonSVR pkg = LIBSVM, kernel = RadialBasis)
    return nothing
end

function param_grid(model, step=1.0; cost=true, epsilon=true)
    # sigma -> gamma (for asin)
    gamma_values = let
        sigma_range = 10.0 .^ (-3:step:3)
        sigma2gamma = sigma -> 1 ./ (2 .* sigma .^ 2)
        sigma2gamma(sigma_range)
    end

    kernel = model.kernel

    if kernel == Kernel.RadialBasis || kernel == Kernel.Sigmoid || kernel == Kernel.Asin || kernel == Kernel.AsinNorm
        gamma = true
        if kernel == Kernel.RadialBasis
            gamma_values = 10 .^ (-3:step:0)
        end
    else
        gamma = false
    end

    return filter(!isnothing, [(
            if (gamma)
                [range(model, :gamma, values=gamma_values)]
            else
                nothing
            end
        ),
        (
            if (epsilon)
                [range(model, :epsilon, values=10 .^ (-5:step:1))]
            else
                nothing
            end
        ),
        (
            if (cost)
                [range(model, :cost, values=10 .^ (-2:step:6))]
            else
                nothing
            end
        )
    ])

end


function TunedModelGridCV(model; nfolds=10, hyper_grid=param_grid(model), args...)
    return TunedModel(
        model=model,
        tuning=Grid(resolution=10),
        resampling=CV(nfolds=nfolds),
        range=hyper_grid,
        measure=rms,
        args...
    )
end
