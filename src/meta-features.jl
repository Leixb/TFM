module MetaFeatures

import ..DataSets: DataSet, unpack

using PyCall
using MLJBase

using DataFrames
using ScientificTypesBase
using StatsBase

using MLJModelInterface
const MMI = MLJModelInterface

"Return a DataFrame with the meta-features extracted from the dataset using the pymfe library."
function extract_features(X, y)
    X = Matrix(X)
    MFE = pyimport("pymfe.mfe").MFE
    mfe = MFE()
    mfe.fit(X, y)
    ft = mfe.extract()

    zip(ft...) |> Dict |> DataFrame
end

function extract_features(ds::DataSet)
    X, y = unpack(ds)
    extract_features(X, y)
end

function extract_features(datasets::Vector{<:DataSet})
    features = map(datasets) do ds
        @info name(ds)
        ft = extract_features(ds)
        ft.dataset = [name(ds)]
        ft
    end

    reduce(vcat, features)
end


@mlj_model struct GeneralMetaFeatureExtractor <: Static
    attr_to_inst::Bool = true # ratio between the number of attributes.
    cat_to_num::Bool = true   # ratio between the number of categorical and numerical attributes.
    freq_class::Bool = true   # relative frequency of each distinct class.
    inst_to_attr::Bool = true # ratio between the number of instances and attributes.
    nr_attr::Bool = true      # total number of attributes.
    nr_bin::Bool = true       # number of binary attributes.
    nr_cat::Bool = true       # number of categorical attributes.
    nr_class::Bool = true     # number of distinct classes.
    nr_inst::Bool = true      # number of instances (rows).
    nr_num::Bool = true       # number of numerical attributes.
    num_to_cat::Bool = true   # ratio between the number of numerical and categorical attributes.
end

function MMI.transform(mfe::GeneralMetaFeatureExtractor, X::DataFrame, y::AbstractVector)
    meta_features = Dict{Symbol,Union{Int,Float64,Vector{Float64}}}()
    if mfe.attr_to_inst
        meta_features[:attr_to_inst] = size(X, 2) / size(X, 1)
    end

    if mfe.cat_to_num
        cat_cols = count(x -> x <: Finite, schema(X).scitypes)
        meta_features[:cat_to_num] = cat_cols / (size(X, 2) - cat_cols)
    end

    if mfe.freq_class
        meta_features[:freq_class] = (countmap(y) |> values |> collect |> sort) ./ length(y)
    end

    if mfe.inst_to_attr
        meta_features[:inst_to_attr] = size(X, 1) / size(X, 2)
    end

    if mfe.nr_attr
        meta_features[:nr_attr] = size(X, 2)
    end

    if mfe.nr_bin
        meta_features[:nr_bin] = count(x -> x <: Finite{2}, schema(X).scitypes)
    end

    if mfe.nr_cat
        meta_features[:nr_cat] = count(x -> x <: Finite, schema(X).scitypes)
    end

    if mfe.nr_class
        meta_features[:nr_class] = length(unique(y))
    end

    if mfe.nr_inst
        meta_features[:nr_inst] = size(X, 1)
    end

    if mfe.nr_num
        meta_features[:nr_num] = count(x -> x <: Continuous, schema(X).scitypes)
    end

    if mfe.num_to_cat
        num_cols = count(x -> x <: Continuous, schema(X).scitypes)
        meta_features[:num_to_cat] = num_cols / (size(X, 2) - num_cols)
    end

    meta_features
end

end
