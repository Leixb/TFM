module TFM

include("./resampling.jl")
include("./datasets.jl")
include("./models.jl")
include("./mse.jl")

export MSE, MeanSquaredError, mse, mean_squared_error

end # module TFM
