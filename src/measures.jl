module Measures

using StatisticalTraits
import MLJBase: @create_aliases, @create_docs, Aggregated, DOC_INFINITE, InfiniteArrMissing, Mean, Measure, detailed_doc_string, mean, metadata_measure, skipinvalid, std, rmse

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

# ----------------------------------------------------------------
# NormalizedMeanSquaredError

struct NormalizedMeanSquaredError <: Aggregated end

metadata_measure(NormalizedMeanSquaredError;
                 instances                = ["nmse", "normalized_mean_squared_error"],
                 target_scitype           = InfiniteArrMissing,
                 prediction_type          = :deterministic,
                 orientation              = :loss,
                 aggregation              = Mean())

const nMSE = NormalizedMeanSquaredError
@create_aliases NormalizedMeanSquaredError

@create_docs(NormalizedMeanSquaredError,
body = """
``\\text{normalized mean squared error} = \\frac{n^{-1}∑ᵢ|yᵢ-ŷᵢ|^2}{var(ŷ)}`` or
``\\text{normalized mean squared error} = \\frac{∑ᵢwᵢ|yᵢ-ŷᵢ|^2}{var(ŷ)∑ᵢwᵢ}``
""",
scitype = DOC_INFINITE)

(::NormalizedMeanSquaredError)(ŷ, y) = mse(ŷ, y) / (std(y)^2)
(::NormalizedMeanSquaredError)(ŷ, y, w) = mse(ŷ, y, w) / (std(y)^2)

export NormalizedMeanSquaredError, nMSE, nmse, normalized_mean_squared_error

# ----------------------------------------------------------------
# NormalizedRootMeanSquaredError

struct NormalizedRootMeanSquaredError <: Aggregated end

metadata_measure(NormalizedRootMeanSquaredError;
                 instances                = ["nrmse", "normalized_root_mean_squared_error"],
                 target_scitype           = InfiniteArrMissing,
                 prediction_type          = :deterministic,
                 orientation              = :loss,
                 aggregation              = RootMeanSquare())

const nRMSE = NormalizedRootMeanSquaredError
@create_aliases NormalizedRootMeanSquaredError

@create_docs(NormalizedRootMeanSquaredError,
body = """
``\\text{normalized root mean squared error} = \\frac{\\sqrt{n^{-1}∑ᵢ|yᵢ-ŷᵢ|^2}}{std(y)}`` or
``\\text{normalized root mean squared error} = \\frac{\\sqrt{\\frac{∑ᵢwᵢ|yᵢ-ŷᵢ|^2}{∑ᵢwᵢ}}}{std(y)}``
""",
scitype = DOC_INFINITE)

(::NormalizedRootMeanSquaredError)(ŷ, y) = rmse(ŷ, y) / std(y)
(::NormalizedRootMeanSquaredError)(ŷ, y, w) = rmse(ŷ, y, w) / std(y)

export NormalizedRootMeanSquaredError, nRMSE, nrmse, normalized_root_mean_squared_error
end # module Measures
