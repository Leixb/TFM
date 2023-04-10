module Resampling

using MLJ
using Random
import MLJBase: train_test_pairs

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
Repeated cross-validation. The `resampling` strategy is repeated `repeats` times. The default is 5 repeats of
2-fold cross-validation.

# Important

This method assumes that the if the `Resampling` object has a field `rng` if and only if it
uses it to initialize the random number generator.
"""
struct RepeatedCV{S<:ResamplingStrategy} <: ResamplingStrategy
    repeats::Int
    resampling::S
    rng::Union{Int,AbstractRNG}

    function RepeatedCV{S}(repeats, rng, args...; kwargs...) where S<:ResamplingStrategy

        if rng isa Integer
            rng = MersenneTwister(rng)
        end
        if rng === nothing
            rng = Random.GLOBAL_RNG
        end
        Random.seed!

        if hasfield(S, :rng)
            resampling = S(args...; kwargs..., rng=rng)
        else
            resampling = S(args...; kwargs...)
        end

        return new(repeats, resampling, rng)
    end
end

function RepeatedCV{S}(; repeats=5, rng=nothing, kwargs...) where S<:ResamplingStrategy
    return RepeatedCV{S}(repeats, rng; kwargs...)
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
        return RepeatedCV{TwoFold}(5, rng)
    end
end

function FiveTwo(; rng=nothing)
    return FiveTwo(rng)
end

end
