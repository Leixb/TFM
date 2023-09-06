#!/usr/bin/env julia

using TFM

using DrWatson

ft = MetaFeatures.extract_features(
    DataSets.all;
    groups=["general", "statistical", "info-theory", "model-based", "landmarking"],
    measure_time="avg",
    random_state=12345
)


@tagsave(datadir("metafeatures", "pymfe.jld2"), Dict("data" => ft))
