#!/usr/bin/env julia

using DataFrames: DataFrame
using MLJ
using LIBSVM.Kernel

EpsilonSVR = @load EpsilonSVR pkg=LIBSVM

function prepare_dataset(df::DataFrame, target::Symbol, ratio=0.9, rng=1234)::Tuple{DataFrame,DataFrame,AbstractVector{Number},AbstractVector{Number}}
    y, X = unpack(df, ==(target); shuffle=false)

    (Xtrain, Xtest), (ytrain, ytest) = partition((X, y), ratio, rng=rng, multi=true)

    return Xtrain, Xtest, ytrain, ytest
end

function run_single(kernel, df, target, ratio=0.9, rng=1234, args...)
    model = ContinuousEncoder() |>
        EpsilonSVR(kernel=kernel, args...)

    Xtrain, Xtest, ytrain, ytest = prepare_dataset(df, target, ratio, rng)

    mach = machine(model, Xtrain, ytrain)
    fit!(mach)

    yhat = predict(mach, Xtest)

    return yhat, ytest
end

function run_gridsearch(kernel, df, target, ratio=0.9, rng=1234, nfolds=5, args...)
    model = EpsilonSVR(kernel=kernel, args...)
    tunned_model = TunedModelGridCV(model; nfolds=nfolds)

    pipeline = ContinuousEncoder() |>
        tunned_model

    Xtrain, Xtest, ytrain, ytest = prepare_dataset(df, target, ratio, rng)

    mach = machine(pipeline, Xtrain, ytrain)
    fit!(mach)

    yhat = predict(mach, Xtest)

    return yhat, ytest, (mach, Xtrain, ytrain, Xtest, ytest)
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


function TunedModelGridCV(model; nfolds=5, hyper_grid=param_grid(model), args...)
    return TunedModel(
        model=model,
        tuning=Grid(resolution=10),
        resampling=CV(nfolds=nfolds),
        range=hyper_grid,
        measure=rms,
        args...
    )
end
