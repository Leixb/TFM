module Models

using MLJ
using MLJBase
using MLJLIBSVMInterface
using LIBSVM: Kernel
using Memoization

import MLJ: partition

using ..DataSets: DataSet, data, target, CategoricalDataSet, RegressionDataSet, MNIST, DelveRegressionDataSet, SolarFlare, ForestFires
import ..Transformers
import ..Utils
import ..Measures: mse

"""
Partition the dataset into training and test sets.

### Example

```julia
using MLJ, TFM
ds = TFM.DataSets.CPU()
(Xtrain, Xtest), (Ytrain, Ytest) = partition(ds)
```
"""
@memoize partition(ds::DataSet; ratio=0.8, shuffle=true, rng=1234, args...) =
    partition(unpack(ds), ratio; shuffle, rng, multi=true, args...)

# ForestFires and SolarFlare have too much zeroes in the target variable,
# so we need to stratify the partition to make sure we have some non-zero
# values in the training and test sets.
@memoize function partition(ds::Union{SolarFlare,ForestFires}; ratio=0.8, shuffle=true, rng=1234, args...)
    X, y = unpack(ds)
    partition((X, y), ratio; shuffle, rng, multi=true, stratify=y, args...)
end


"""
# Base models

The models used for each type of dataset.
"""
EpsilonSVR = @load EpsilonSVR pkg = LIBSVM verbosity = 0
SVC = @load SVC pkg = LIBSVM verbosity = 0

basemodel(::DataSet) = EpsilonSVR # default to regression

basemodel(::RegressionDataSet) = EpsilonSVR
basemodel(::CategoricalDataSet) = SVC

basemodel(::DelveRegressionDataSet) = EpsilonSVR
basemodel(::MNIST) = SVC

"""
# Create an MLJ pipeline for the given dataset.

The pipeline is a `ContinuousEncoder` followed by a `Standardizer` and
a `TransformedTargetModel` which applies the `Standardizer` to
the target variable also. The base model is an `EpsilonSVR` with the
given kernel and arguments.
"""
function pipeline(ds::DataSet; kernel=Kernel.RadialBasis, gamma=0.5, args...)
    common = ContinuousEncoder() |> Standardizer()

    if kernel in [Kernel.Acos0, Kernel.Acos1, Kernel.Acos2]
        common = common |> Transformers.Multiplier(factor=sqrt(Utils.gamma2sigma(gamma)))
    end

    if ds isa RegressionDataSet || ds isa DelveRegressionDataSet
        return common |>
               TransformedTargetModel(basemodel(ds)(; kernel, gamma, args...); transformer=Standardizer())
    end

    common |>
    basemodel(ds)(; kernel, gamma, args...)
end

"""

# Specific pipeline for MNIST

We do not need to one-hot encode or standardize anything.
"""
pipeline(ds::MNIST; kernel=Kernel.RadialBasis, args...) = basemodel(ds)(; kernel, args...)

end
