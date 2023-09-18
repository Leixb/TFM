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
    TwoFoldZeroStratified <: ResamplingStrategy

Like `TwoFold`, but stratifies the data on the zero values of the target variable.
"""
struct TwoFoldZeroStratified <: ResamplingStrategy
    rng::Union{Int,AbstractRNG}

    function TwoFoldZeroStratified(rng)
        return new(rng)
    end
end

function TwoFoldZeroStratified(; rng=nothing)
    if rng isa Integer
        rng = MersenneTwister(rng)
    end
    if rng === nothing
        rng = Random.GLOBAL_RNG
    end
    Random.seed!
    return TwoFoldZeroStratified(rng)
end

function train_test_pairs(twofold::TwoFoldZeroStratified, rows)
    y = [row[end] for row in rows]
    train, test = partition(rows, 0.5, shuffle=true, rng=twofold.rng, stratify=y .== 0)
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

    function RepeatedCV{S}(repeats, rng, args...; kwargs...) where {S<:ResamplingStrategy}

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

function RepeatedCV{S}(; repeats=5, rng=nothing, kwargs...) where {S<:ResamplingStrategy}
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

"""
    FiveTwoZeroStratified <: ResamplingStrategy

5x2 cross-validation with stratification on the zero values of the target variable.
"""
struct FiveTwoZeroStratified <: ResamplingStrategy
    function FiveTwoZeroStratified(rng)
        return RepeatedCV{TwoFoldZeroStratified}(5, rng)
    end
end

function FiveTwoZeroStratified(; rng=nothing)
    return FiveTwoZeroStratified(rng)
end

# Functions to print resampling strategies inside filenames
Base.show(io::IO, resampling::CV) = print(io, "CV-$(resampling.nfolds)")
Base.show(io::IO, resampling::StratifiedCV) = print(io, "SCV-$(resampling.nfolds)")
Base.show(io::IO, resampling::Holdout) = print(io, "Holdout-$(resampling.fraction_train)")
Base.show(io::IO, ::TwoFold) = print(io, "TwoFold")
Base.show(io::IO, ::FiveTwo) = print(io, "5x2")
Base.show(io::IO, ::TwoFoldZeroStratified) = print(io, "TwoFoldZS")
Base.show(io::IO, ::FiveTwoZeroStratified) = print(io, "5x2ZS")
Base.show(io::IO, resampling::RepeatedCV{TwoFold}) = print(io, resampling.repeats, "x2")
Base.show(io::IO, resampling::RepeatedCV) = print(io, "$(resampling.repeats)x", resampling.resampling)

end
