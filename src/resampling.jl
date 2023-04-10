module Resampling

using MLJ
using Random

export TwoFold, FiveTwo, RepeatedCV

"""
    TwoFold <: ResamplingStrategy

2-fold cross-validation. The data is split into two parts, and the model is trained
on one part and tested on the other. The process is repeated with the roles of the
two parts reversed.

## Example

```julia
cv = TwoFold(; rng=1234)
```
"""
struct TwoFold <: ResamplingStrategy
    rng::Union{Int,AbstractRNG}

    function TwoFold(rng)
        return new(rng)
    end
end

function TwoFold(; rng=nothing)
    if rng isa Integer
        rng = MersenneTwister(rng)
    end
    if rng === nothing
        rng = Random.GLOBAL_RNG
    end
    Random.seed!
    return TwoFold(rng)
end

function train_test_pairs(twofold::TwoFold, rows)
    train, test = partition(rows, 0.5, shuffle=true, rng=twofold.rng)
    return [(train, test), (test, train)]
end

"""
    RepeatedCV(repeats=5, resampling=TwoFold())

Repeated cross-validation. The `resampling` strategy is repeated `repeats` times. The default is 5 repeats of
2-fold cross-validation.

# Important

The `rng` of the internal `resampling` strategy has to be shared by all repeats, as such
you must make sure they use the same random number generator object.

This means, that `rng` should be initialized outside of the `resampling` object creation.
"""
struct RepeatedCV{S<:ResamplingStrategy} <: ResamplingStrategy
    repeats::Int
    resampling::S

    function RepeatedCV(repeats, resampling)

        if hasproperty(resampling, :rng) && resampling.rng isa Integer
            @warn "Using integer as rng for resampling strategy. This will probably not work with RepeatedCV."
        end

        return new{typeof(resampling)}(repeats, resampling)
    end
end

function RepeatedCV(; repeats=5, resampling=TwoFold())
    return RepeatedCV(repeats, resampling)
end

function train_test_pairs(repeatedcv::RepeatedCV, rows)
    return vcat(map(_ -> train_test_pairs(repeatedcv.resampling, rows), 1:repeatedcv.repeats)...)
end

"""
    FiveTwo <: ResamplingStrategy

5x2 cross-validation

Dietterich (1998) https://doi.org/10.1162/089976698300017197
"""
struct FiveTwo <: ResamplingStrategy
    function FiveTwo(rng)
        return RepeatedCV(5, TwoFold(), rng)
    end
end

function FiveTwo(; rng=nothing)
    return FiveTwo(rng)
end

end
