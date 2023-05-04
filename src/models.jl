module Models

using MLJ
using MLJBase
using MLJLIBSVMInterface
using LIBSVM: Kernel

import MLJ: unpack, partition

using ..DataSets: DataSet, data, target, CategoricalDataSet, RegressionDataSet, CPU, MNIST, Cancer, Triazines, Ailerons, Elevators
import ..Transformers
import ..Utils
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
function pipeline(ds::DataSet; kernel=Kernel.RadialBasis, gamma=0.5, args...)
    common = ContinuousEncoder() |>
    Standardizer()

    if kernel in [Kernel.Acos0, Kernel.Acos1, Kernel.Acos2]
        common = common |> Transformers.Multiplier(factor=sqrt(Utils.gamma2sigma(gamma)))
    end

    common |>
        TransformedTargetModel(basemodel(ds)(;kernel, gamma, args...); transformer=Standardizer())
end

drop_columns(features::Symbol...) = FeatureSelector(features=[features...], ignore=true)
select_columns(features::Symbol...) = FeatureSelector(features=[features...], ignore=false)

pipeline(ds::CPU; args...) =
    drop_columns(:Model, :Vendor, :PRP) |>
    invoke(pipeline, Tuple{DataSet}, ds; args...)


# Column1 is just an index
pipeline(ds::Cancer; args...) =
    drop_columns(:Column1) |>
    invoke(pipeline, Tuple{DataSet}, ds; args...)

# Triazines has two columns with all 0
pipeline(ds::Triazines; args...) =
    drop_columns(:p5_flex, :p5_h_doner) |>
    invoke(pipeline, Tuple{DataSet}, ds; args...)

# Ailerons has a column with all 0 and one with only 1 value different from 0
pipeline(ds::Ailerons; args...) =
    select_columns(:climbRate, :Sgz, :p, :q, :curPitch, :curRoll, :absRoll) |>
    invoke(pipeline, Tuple{DataSet}, ds; args...)

# Elevators has a column with all 0
pipeline(ds::Elevators; args...) =
    select_columns(:climbRate, :Sgz, :p, :q, :curRoll, :absRoll) |>
    invoke(pipeline, Tuple{DataSet}, ds; args...)
"""

# Specific pipeline for MNIST

We do not need to one-hot encode or standardize anything.
"""
pipeline(ds::MNIST; kernel=Kernel.RadialBasis, args...) = basemodel(ds)(;kernel, args...)

end
