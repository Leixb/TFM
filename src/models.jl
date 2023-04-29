module Models

using MLJ
using MLJBase
using MLJLIBSVMInterface
using LIBSVM: Kernel

import MLJ: unpack, partition

using ..DataSets: DataSet, data, target, CategoricalDataSet, RegressionDataSet, CPU, MNIST, Cancer
import ..Transformers: TopCatTransformer
import ..Measures: mse

"""
Split the dataset into features and target.
"""
unpack(ds::DataSet; args...) = unpack(data(ds), !=(target(ds)); args...)

"""
MNIST is already in the right format.
"""
unpack(ds::MNIST) = data(ds)

"""
Partition the dataset into training and test sets.

### Example

```julia
using MLJ, TFM
ds = TFM.DataSets.CPU()
(Xtrain, Xtest), (Ytrain, Ytest) = partition(ds)
```
"""
partition(ds::DataSet; ratio=0.8, shuffle=true, rng=1234, args...) =
    partition(unpack(ds), ratio; shuffle, rng, multi=true, args...)


"""
# Base models

The models used for each type of dataset.
"""
EpsilonSVR = @load EpsilonSVR pkg=LIBSVM verbosity=0
SVC = @load SVC pkg=LIBSVM verbosity=0
basemodel(::RegressionDataSet) = EpsilonSVR
basemodel(::CategoricalDataSet) = SVC

"""
# Create an MLJ pipeline for the given dataset.

The pipeline is a `ContinuousEncoder` followed by a `Standardizer` and
a `TransformedTargetModel` which applies the `Standardizer` to
the target variable also. The base model is an `EpsilonSVR` with the
given kernel and arguments.
"""
pipeline(ds::DataSet; kernel=Kernel.RadialBasis, args...) =
    ContinuousEncoder() |>
    Standardizer() |>
    TransformedTargetModel(basemodel(ds)(;kernel, args...); transformer=Standardizer())

"""

# Specific pipeline for CPU dataset

- `:Vendor` is encoded by keeping the top 3 most frequent values and setting
the rest to `OTHER`.
- `:Model` is dropped because it is nearly unique.

"""
pipeline(ds::CPU; args...) =
    FeatureSelector(features=[:Model], ignore=true) |>
    TopCatTransformer(n=3) |>
    invoke(pipeline, Tuple{DataSet}, ds; args...)

pipeline(ds::Cancer; args...) =
    FeatureSelector(features=[:Column1], ignore=true) |>
    invoke(pipeline, Tuple{DataSet}, ds; args...)

"""

# Specific pipeline for MNIST

We do not need to one-hot encode or standardize anything.
"""
pipeline(ds::MNIST; kernel=Kernel.RadialBasis, args...) = basemodel(ds)(;kernel, args...)


function tuned_model(model; step=1.0, resampling=CV(nfolds=10), measure=mse, args...)
    gamma_values = let
        sigma_range = 10.0 .^ (-3:1.0:3)
        sigma2gamma = sigma -> 1 ./ (2 .*sigma.^2)
        sigma2gamma(sigma_range)
    end

    if model isa MLJBase.DeterministicPipeline
        inner_model = model.transformed_target_model_deterministic.model
    else
        inner_model = model
    end

    param_range = [
        range(model, :(transformed_target_model_deterministic.model.cost), values = 10 .^ (-2:step:6))
    ]

    if inner_model isa MLJLIBSVMInterface.EpsilonSVR
        param_range = vcat(param_range,
            range(model, :(transformed_target_model_deterministic.model.epsilon), values = 10 .^ (-5:step:1)),
        )
    elseif inner_model isa MLJLIBSVMInterface.SVC
        measure = accuracy
    else
        error("Model $(typeof(inner_model)) not supported")
    end

    if inner_model.kernel == Kernel.RadialBasis
        param_range = vcat(param_range,
            range(model, :(transformed_target_model_deterministic.model.gamma), values = 10 .^ (-3:step:0)),
        )
    elseif inner_model.kernel == Kernel.Asin || inner_model.kernel == Kernel.AsinNorm
        param_range = vcat(param_range,
            range(model, :(transformed_target_model_deterministic.model.gamma), values = gamma_values),
        )
    else
        error("Kernel $(inner_model.kernel) not supported")
    end

    TunedModel(;model, resampling, tuning = Grid(), range=param_range, measure, args...)
end

end
