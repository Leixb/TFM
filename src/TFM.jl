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

include("./transformers.jl")
export Transformers

include("./resampling.jl")
export Resampling

include("./measures.jl")
export Measures

include("./datasets.jl")
export DataSets

include("./models.jl")
export Models

include("./experiments.jl")
export Experiments

# HACK: Export `mse` like this so that deserialization works with machines
# that were saved before the Measures module was created.
import .Measures
function mse(ŷ, y) return Measures.mse(ŷ, y) end
export mse

@reexport import .Transformers: TopCatTransformer, TopCatResult

end # module TFM
