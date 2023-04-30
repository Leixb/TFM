module Measures

using MLJBase
using StatisticalTraits

import MLJBase: Measure, Aggregated, @create_aliases, @create_docs, DOC_INFINITE, detailed_doc_string, InfiniteArrMissing

struct MeanSquare <: Aggregated end
(::MeanSquare)(v) = mean(skipinvalid(v).^2)

struct MeanSquaredError <: Aggregated end

metadata_measure(MeanSquaredError;
                 instances                = ["mse", "mean_squared_error"],
                 target_scitype           = InfiniteArrMissing,
                 prediction_type          = :deterministic,
                 orientation              = :loss,
                 aggregation              = MeanSquare())

call(::MeanSquaredError, ŷ, y) = (y .- ŷ).^2 |> skipinvalid |> mean
call(::MeanSquaredError, ŷ, y, w) = (y .- ŷ).^2 .* w |> skipinvalid |> mean

const MSE = MeanSquaredError
@create_aliases MeanSquaredError

@create_docs(MeanSquaredError,
body=
"""
``\\text{mean squared error} = n^{-1}∑ᵢ|yᵢ-ŷᵢ|^2`` or
``\\text{mean squared error} = \\frac{∑ᵢwᵢ|yᵢ-ŷᵢ|^2}{∑ᵢwᵢ}``
""",
scitype=DOC_INFINITE)

export MeanSquare, MeanSquaredError, MSE, mse, mean_squared_error

end # module Measures
