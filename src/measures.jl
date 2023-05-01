module Measures

using StatisticalTraits
import MLJBase: @create_aliases, @create_docs, Aggregated, DOC_INFINITE, InfiniteArrMissing, Mean, Measure, detailed_doc_string, mean, metadata_measure, skipinvalid

# ----------------------------------------------------------------
# MeanSquaredError

struct MeanSquaredError <: Aggregated end

metadata_measure(MeanSquaredError;
                 instances                = ["mse", "mean_squared_error"],
                 target_scitype           = InfiniteArrMissing,
                 prediction_type          = :deterministic,
                 orientation              = :loss,
                 aggregation              = Mean())

const MSE = MeanSquaredError
@create_aliases MeanSquaredError

@create_docs(MeanSquaredError,
body = """
``\\text{mean squared error} = n^{-1}∑ᵢ|yᵢ-ŷᵢ|^2`` or
``\\text{mean squared error} = \\frac{∑ᵢwᵢ|yᵢ-ŷᵢ|^2}{∑ᵢwᵢ}``
""",
scitype = DOC_INFINITE)

(::MeanSquaredError)(ŷ, y) = (y .- ŷ) .^ 2 |> skipinvalid |> mean
(::MeanSquaredError)(ŷ, y, w) = (y .- ŷ) .^ 2 .* w |> skipinvalid |> mean

export MeanSquaredError, MSE, mse, mean_squared_error

"""
Display the name of MLJMeasures using the first instance alias.
Useful for printing the name of the measure in a table or in a filename.
"""
Base.show(io::IO, measure::Measure) = print(io, instances(typeof(measure))[1])

end # module Measures
