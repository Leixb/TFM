using MLJBase
using Tables
using CategoricalArrays
using StatsBase
using MLJModelInterface

const MMI = MLJModelInterface

"""
Takes all categorical columns and only uses the top n categories for each. The
rest are replaced with a new category called "OTHER".

# Arguments
 - `features`: a vector of features to be encoded. If empty, all categorical
   features are encoded.
 - `n`: the number of top categories to be used. If negative or nothing, all
  categories are used.
 - `cutoff`: if cutoff is an integer, take all categories with count >= cutoff,
   if it's a float (between 0 and 1), take all with higher relative frequency.
 - `other`: the name of the new category for the rest of the categories.
 - `ordered_factor`: if true, consider also ordered factors.
 - `ignore`: if true, ignore the features specified in `features` and encode
   all categorical features except those.
"""
@mlj_model mutable struct TopCatTransformer <: Unsupervised
    features::Vector{Symbol} = Symbol[]
    n::Union{Int,Nothing} = 5
    cutoff::Union{Integer,AbstractFloat,Nothing} = nothing # if cutoff is an integer, take all categories with count >= cutoff, if it's a float (between 0 and 1), take all with higher relative frequency.
    other::String = "OTHER"
    ordered_factor::Bool = true
    ignore::Bool = false
end

struct TopCatResult <: MMI.MLJType
    top_n_given_feature::Dict{Symbol,CategoricalArray}
end

function MLJBase.fit(transformer::TopCatTransformer, verbosity::Int, X)
    all_features = Tables.schema(X).names # a tuple not vector

    if isempty(transformer.features)
        specified_features = collect(all_features)
    else
        if transformer.ignore
            specified_features = filter(all_features |> collect) do ftr
                !(ftr in transformer.features)
            end
        else
            specified_features = transformer.features
        end
    end

    allowed_scitypes = ifelse(transformer.ordered_factor, Finite, Multiclass)
    top_n_given_feature = Dict{Symbol,AbstractArray{AbstractString}}()
    col_scitypes = schema(X).scitypes
    # apply on each feature
    for j in eachindex(all_features)
        ftr = all_features[j]
        col = MMI.selectcols(X, j)
        T = col_scitypes[j]
        if T <: allowed_scitypes && ftr in specified_features
            freqs = collect(StatsBase.countmap(col))

            n = transformer.n
            # if n is negative or nothing, take all categories
            if transformer.n < 0 || transformer.n === nothing
                n = length(freqs)
            end

            sorted = sort(freqs, by=x -> (-x[2], x[1]))
            topn = Iterators.take(sorted, n)
            if transformer.cutoff !== nothing
                if transformer.cutoff isa Integer
                    topn = Iterators.takewhile(topn) do (ref, count)
                        count >= transformer.cutoff
                    end
                else # transformer.cutoff isa AbstractFloat
                    @assert 0 <= transformer.cutoff <= 1 "cutoff must be between 0 and 1 (or an integer)"
                    topn = Iterators.takewhile(topn) do (ref, count)
                        count / length(col) >= transformer.cutoff
                    end
                end
            end

            topn = Iterators.map(first, topn)
            if verbosity > 0
                @debug "Top $(transformer.n) categories for feature :$ftr are $(topn)"
            end
            top_n_given_feature[ftr] = collect(topn)
        end
    end

    fitresult = TopCatResult(top_n_given_feature,)

    report = (features_to_be_encoded=collect(keys(top_n_given_feature)),
        top_n_given_feature=top_n_given_feature,)
    cache = nothing

    return fitresult, cache, report
end

function MMI.transform(transformer::TopCatTransformer, fitresult, X)
    features = Tables.schema(X).names     # tuple not vector

    d = fitresult.top_n_given_feature

    features_to_be_transformed = keys(d)
    for ftr in features
        if !(ftr in features_to_be_transformed)
            continue
        end
        col = MMI.selectcols(X, ftr)
        others = setdiff(Set(MMI.classes(col)), Set(d[ftr])) |> collect
        recode!(col, others => transformer.other)
    end

    return X
end
