module TFM

using Reexport

@reexport using DataFrames, MLJ
@reexport import LIBSVM

"""
Root type for all custom types in the TFM module.

Note: this is only used for types that are not part of the MLJ framework
or any other package.
"""
abstract type TFMType end

include("./utils.jl")
export Utils

include("./transformers.jl")
export Transformers
@reexport import .Transformers: TopCatTransformer, TopCatResult

include("./resampling.jl")
export Resampling

include("./measures.jl")
export Measures

include("./datasets.jl")
export DataSets

include("./meta-features.jl")
export MetaFeatures

include("./models.jl")
export Models

include("./experiments.jl")
export Experiments

include("./benchmark.jl")

# only if we are in an interactive session
include("./plots.jl")

# HACK: Export `mse` like this so that deserialization works with machines
# that were saved before the Measures module was created.
import .Measures
function mse(ŷ, y)
    return Measures.mse(ŷ, y)
end
export mse

@static if false
    # HACK: Trick LSP into thinking that files outside `src` are part of the
    # module.
    # see: https://discourse.julialang.org/t/lsp-missing-reference-woes/98231/11
    module LspLink
    include("../scripts/produce_all.jl")
    include("../scripts/plots.jl")
    include("../scripts/save_datasets.jl")
    include("../scripts/extract_features.jl")
    include("../scripts/produce_precomp_vs_native.jl")
    end
end

end # module TFM
