module Models

using MLJ
using LIBSVM: Kernel

using ..DataSets: DataSet, data, target, CategoricalDataSet, RegressionDataSet, CPU, MNIST
import ..TopCatTransformer


"""
Split the dataset into features and target.
"""
unpack(ds::DataSet; args...) = unpack(data(ds), !=(target(ds)); args...)

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
basemodel(::RegressionDataSet) = @load EpsilonSVR pkg=LIBSVM verbosity=0
basemodel(::CategoricalDataSet) = @load SVC pkg=LIBSVM verbosity=0

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

"""

# Specific pipeline for MNIST

We do not need to one-hot encode or standardize anything.
"""
pipeline(ds::MNIST; kernel=Kernel.RadialBasis, args...) = basemodel(ds)(;kernel, args...)

end
