using MLJBase
using Tables
using CategoricalArrays
using StatsBase
using MLJModelInterface

const MMI = MLJModelInterface

# Take the top n categories of each feature and replace the rest with "OTHER"
@mlj_model mutable struct TopCatTransformer <: Unsupervised
    features::Vector{Symbol}   = Symbol[]
    n::Int                     = 5
    other::String              = "OTHER"
    ordered_factor::Bool       = true
    ignore::Bool               = false
end

struct TopCatResult <: MMI.MLJType
    top_n_given_feature::Dict{Symbol, CategoricalArray}
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
    top_n_given_feature = Dict{Symbol, CategoricalArray}()
    col_scitypes = schema(X).scitypes
    # apply on each feature
    for j in eachindex(all_features)
        ftr = all_features[j]
        col = MMI.selectcols(X,j)
        T = col_scitypes[j]
        if T <: allowed_scitypes && ftr in specified_features
            freqs = collect(StatsBase.countmap(col))

            n = transformer.n
            # if n is negative, take all categories
            if transformer.n < 0
                n = length(freqs)
            elseif transformer.n > length(freqs)
                n = length(freqs)
                @warn "n ($(transformer.n)) is greater than the number of categories for feature :$ftr. Setting n to $(n)"
            end

            topn = sort(freqs, by = x -> (-x[2], x[1]))[1:n]
            topn = map(collect(topn)) do (ref, _) ref end
            if verbosity > 0
                @debug "Top $(transformer.n) categories for feature :$ftr are $(topn)"
            end

            top_n_given_feature[ftr] = topn
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
