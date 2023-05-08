module Models

using MLJ
using MLJBase
using MLJLIBSVMInterface
using LIBSVM: Kernel

import MLJ: partition

using ..DataSets: DataSet, data, target, CategoricalDataSet, RegressionDataSet, MNIST
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
function pipeline(ds::DataSet; kernel=Kernel.RadialBasis, gamma=0.5, args...)
    common = ContinuousEncoder() |>
    Standardizer()

    if kernel in [Kernel.Acos0, Kernel.Acos1, Kernel.Acos2]
        common = common |> Transformers.Multiplier(factor=sqrt(Utils.gamma2sigma(gamma)))
    end

    common |>
        TransformedTargetModel(basemodel(ds)(;kernel, gamma, args...); transformer=Standardizer())
end

"""

# Specific pipeline for MNIST

We do not need to one-hot encode or standardize anything.
"""
pipeline(ds::MNIST; kernel=Kernel.RadialBasis, args...) = basemodel(ds)(;kernel, args...)

end
