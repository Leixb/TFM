module MetaFeatures

import ..DataSets: DataSet, unpack, MNIST, Delve

using PyCall
using DataFrames

export extract_features

# TODO: Choose subset of meta-features to use

"Return a DataFrame with the meta-features extracted from the dataset using the pymfe library."
function extract_features(X, y, args...; features::Vector{String}=String[], kwargs...)
    X = Matrix(X)
    MFE = pyimport("pymfe.mfe").MFE

    blacklist = [
        "lh_trace",
        "cor",
        "cov",
        "can_cor",
        "nr_disc",
        "p_trace",
        "roy_root",
        "w_lambda",
    ]

    if isempty(features)
        features = MFE.valid_metafeatures()
    end

    features = filter(features) do mf
        !in(mf, blacklist)
    end

    mfe = MFE(args...; features, kwargs...)

    mfe.fit(X, y, precomp_groups=["general", "model-based", "landmarking"], show_warnings=false, verbose=2)
    ft = mfe.extract(verbose=2)

    zip(ft...) |> Dict |> DataFrame
end

function extract_features(ds::DataSet, args...; kwargs...)
    X, y = unpack(ds)
    extract_features(X, y, args...; kwargs...)
end

function extract_features(ds::MNIST, args...; kwargs...)
    X, y = unpack(ds)
    X = DataFrame(X)
    extract_features(X, y, args...; kwargs...)
end

function extract_features(datasets::Vector{<:DataSet}, args...; kwargs...)
    features = map(datasets) do ds
        @info string(ds)
        ft = extract_features(ds, args...; kwargs...)
        ft.dataset = [string(ds)]

        ft
    end

    vcat(features..., cols=:union)
end

end
