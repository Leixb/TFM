module TFM

using Reexport

@reexport using DataFrames, MLJ

include("./transformers.jl")
export Transformers

include("./resampling.jl")
export Resampling

include("./measures.jl")
export Measures

# HACK: Export `mse` like this so that deserialization works with machines
# that were saved before the Measures module was created.
function mse(ŷ, y) return Measures.mse(ŷ, y) end
export mse

include("./datasets.jl")
export DataSets

include("./models.jl")
export Models

include("./experiments.jl")
export Experiments

end # module TFM
