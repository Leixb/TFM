using MLJBase
using StatisticalTraits

abstract type Measure <: MLJType end
abstract type Aggregated <: Measure end

struct MeanSquaredError <: Aggregated end

const MSE = MeanSquaredError

call(::MeanSquaredError, ŷ, y) = (y .- ŷ).^2 |> skipinvalid |> mean
call(::MeanSquaredError, ŷ, y, w) = (y .- ŷ).^2 .* w |> skipinvalid |> mean

function mse(ŷ, y)
    return call(MeanSquaredError(), ŷ, y)
end

function mse(ŷ, y, w)
    return call(MeanSquaredError(), ŷ, y, w)
end

const mean_squared_error = mse

struct MeanSquare <: Aggregated end
(::MeanSquare)(v) = mean(skipinvalid(v).^2)

metadata_measure(MeanSquaredError;
                 instances                = ["mse", "mean_squared_error"],
                 prediction_type          = :deterministic,
                 orientation              = :loss,
                 aggregation              = MeanSquare())
