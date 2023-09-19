#!/usr/bin/env julia

"""
Module to specify all datasets used in the project.

This should contain the methods necessary to read and do minimal
pre-processing. For models, see `models.jl`.
"""
module DataSets

import CSV
using DataFrames
using LIBSVM: Kernel
using CategoricalArrays: CategoricalArray
using Random
using Memoization

import ..TFMType

using MLJ
import MLJ: unpack

using MLDatasets: MNIST as MNISTData

"""
DataSet is the top abstract type that provides the interface for reading
and preprocessing a dataset.
"""
abstract type DataSet <: TFMType end

function Base.show(io::IO, ds::DataSet)
    type_name = string(typeof(ds))
    name = split(type_name, '.') |> last
    print(io, name)
end

Base.show(io::IO, ::MIME"text/plain", ds::DataSet) = print(io, typeof(ds), "()")

# These methods should only be implemented if the dataset is not
# in a proper CSV format.
data(ds::DataSet) = raw_data(ds) |> preprocess(ds)
raw_data(ds::DataSet) = read_data(ds; select=select_columns(ds), drop=drop_columns(ds))
read_data(ds::DataSet; kwargs...) = CSV.read(path(ds), DataFrame; header=header(ds), kwargs...)

(ds::DataSet)() = unpack(ds)

# These methods should be implemented for each dataset
target(ds::DataSet) = error("target not implemented for $(typeof(ds))")
path(ds::DataSet) = error("path not implemented for $(typeof(ds))")

# Split dataset into features and target (X, y)
@memoize unpack(ds::DataSet; args...) = unpack(data(ds), !=(target(ds)); args...)

# The following methods are optional, but highly recommended
header(ds::DataSet) = false
preprocess(::DataSet) = identity

# Which columns to select/drop
select_columns(::DataSet) = nothing
drop_columns(::DataSet) = nothing

# Metadata (optional)

# url(ds::DataSet) returns the URL of the dataset (where it was downloaded from)
url(ds::DataSet) = error("url not implemented for $(typeof(ds))")

# doi(ds::DataSet) returns the DOI of the relevant paper that uses the dataset.
doi(::DataSet) = nothing

"List of all datasets defined in the module"
all = Vector{DataSet}()

"""
# Helper macro to declare datasets

The macro takes the following arguments:

- `type`: the type of the dataset (e.g. `Large`, `Small`)
- `name`: the name of the dataset (e.g. `Abalone`)
- `path`: the path to the dataset (e.g. `datadir("abalone")`)
- `header`: An integer to say which row has the header, or an
array of strings to specify the header manually (`false` to generate
a header automatically)
- `target`: the name of the target variable (e.g. `:Rings`).

### Important

The compilation will fail if the target is not part of the header.

"""
macro dataset(type, name, path, header, target)
    lowername = Symbol(lowercase(string(name)))
    # if header is an array, check that target is in it
    if eval(header) isa AbstractArray
        @assert eval(target) in eval(header) "target not in header"
    end
    esc(quote
        struct $name <: $type end
        const $lowername = $name()
        push!(all, $lowername)

        path(::$name) = $path
        header(::$name) = $header
        target(::$name) = $target
    end)
end

# DataSets that are used in Frenay and Verleysen (2016)
abstract type CategoricalDataSet <: DataSet end
abstract type RegressionDataSet <: DataSet end

is_regression(::CategoricalDataSet) = false
is_regression(::RegressionDataSet) = true

preprocess(ds::CategoricalDataSet) = X -> coerce(X, target(ds) => Multiclass)

abstract type Frenay <: RegressionDataSet end
doi(::Frenay) = "10.1016/j.neucom.2010.11.037"

# DataSet relative size according to Frenay and Verleysen (2016)
abstract type Large <: Frenay end
abstract type Small <: Frenay end

datasetdir(path...) = joinpath(ENV["DATASETS"], path...)

################################################################################
# Abalone (ID=1) 1994
#-------------------------------------------------------------------------------
# - 4177 instances
# - 8 attributes
#-------------------------------------------------------------------------------
# Predict the age of abalone from physical measurements
#-------------------------------------------------------------------------------
# # Creators
#
#  - Warwick Nash ()
#  - Tracy Sellers ()
#  - Simon Talbot ()
#  - Andrew Cawthorn ()
#  - Wes Ford ()
#
#-------------------------------------------------------------------------------
# # Attribute Information
#
# - Sex: (Categorical Feature) M, F, and I (infant)
# - Length: (Continuous Feature) Longest shell measurement
# - Diameter: (Continuous Feature) perpendicular to length
# - Height: (Continuous Feature) with meat in shell
# - Whole_weight: (Continuous Feature) whole abalone
# - Shucked_weight: (Continuous Feature) weight of meat
# - Viscera_weight: (Continuous Feature) gut weight (after bleeding)
# - Shell_weight: (Continuous Feature) after being dried
# - Rings: (Integer Target) +1.5 gives the age in years
#
################################################################################

@dataset Large Abalone datasetdir("abalone") [
    :Sex, :Length, :Diameter, :Height, :Whole_weight, :Shucked_weight, :Viscera_weight, :Shell_weight, :Rings
] :Rings
preprocess(ds::Abalone) = X -> coerce(X, target(ds) => Continuous, :Sex => Multiclass)

url(::Abalone) = "https://archive.ics.uci.edu/dataset/1/abalone"
doi(::Abalone) = "10.24432/C55C7W"

################################################################################
# Ailerons
################################################################################

@dataset Large Ailerons datasetdir("ailerons", "ailerons.data") [
    :climbRate, :Sgz, :p, :q, :curPitch, :curRoll, :absRoll, :diffClb, :diffRollRate, :diffDiffClb,
    :SeTime1, :SeTime2, :SeTime3, :SeTime4, :SeTime5, :SeTime6, :SeTime7,
    :SeTime8, :SeTime9, :SeTime10, :SeTime11, :SeTime12, :SeTime13, :SeTime14,
    :diffSeTime1, :diffSeTime2, :diffSeTime3, :diffSeTime4, :diffSeTime5, :diffSeTime6, :diffSeTime7,
    :diffSeTime8, :diffSeTime9, :diffSeTime10, :diffSeTime11, :diffSeTime12, :diffSeTime13, :diffSeTime14,
    :alpha, :Se, :goal,
] :goal
select_columns(::Ailerons) = [:climbRate, :Sgz, :p, :q, :curPitch, :curRoll, :absRoll, :goal]

url(::Ailerons) = "https://www.dcc.fc.up.pt/~ltorgo/Regression/ailerons.tgz"

################################################################################
# Wisconsin Breast Cancer (Prognostic)
################################################################################

@dataset Small Cancer datasetdir("cancer") false :Column3
# Drop ID and diagnosis
drop_columns(::Cancer) = [:Column1, :Column2]

url(::Cancer) = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
doi(::Cancer) = "10.24432/C5GK50"

################################################################################
# CompActs
################################################################################

@dataset Large CompActs datasetdir("compActs", "cpu_act.data") [
    :lread, :lwrite, :scall, :sread, :swrite, :fork, :exec, :rchar, :wchar,
    :pgout, :ppgout, :pgfree, :pgscan, :atch, :pgin, :ppgin, :pflt, :vflt,
    :runqsz, :freemem, :freeswap, :usr
] :usr

url(::CompActs) = "https://www.dcc.fc.up.pt/~ltorgo/Regression/compact.tar.gz"

################################################################################
# CPU
################################################################################

@dataset Small CPU datasetdir("cpu") [
    :Vendor, :Model, :MYCT, :MMIN, :MMAX, :CACH, :CHMIN, :CHMAX, :PRP, :ERP
] :ERP
# Frenay and Verleysen (2011) do not specify the target, but we assume it is ERP.
# PRP is dropped since it is highly correlated with ERP and they only use 6
# features in their experiments.
drop_columns(::CPU) = [:Model, :Vendor, :PRP]


url(::CPU) = "https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data"

################################################################################
# Elevators
################################################################################

@dataset Large Elevators datasetdir("elevators", "elevators.data") [
    :climbRate, :Sgz, :p, :q, :curRoll, :absRoll, :diffClb, :diffRollRate, :diffDiffClb,
    :SaTime1, :SaTime2, :SaTime3, :SaTime4, :diffSaTime1, :diffSaTime2, :diffSaTime3, :diffSaTime4,
    :Sa, :Goal
] :Goal
select_columns(::Elevators) = [:climbRate, :Sgz, :p, :q, :curRoll, :absRoll, :Goal]

url(::Elevators) = "https://www.dcc.fc.up.pt/~ltorgo/Regression/elevators.tgz"

################################################################################
# Stock
################################################################################

@dataset Small Stock datasetdir("stock", "stock.data") [
    :Company1, :Company2, :Company3, :Company4, :Company5,
    :Company6, :Company7, :Company8, :Company9, :Company10,
] :Company10
function raw_data(ds::Stock)
    CSV.read(path(ds), DataFrame; header=header(ds),
        delim="\t",
        ignorerepeated=true
    )
end

url(::Stock) = "https://www.dcc.fc.up.pt/~ltorgo/Regression/stock.tgz"

################################################################################
# Triazines
################################################################################

@dataset Small Triazines datasetdir("triazines", "triazines.data") [
    :p1_polar, :p1_size, :p1_flex, :p1_h_doner, :p1_h_acceptor, :p1_pi_doner, :p1_pi_acceptor, :p1_polarisable, :p1_sigma, :p1_branch,
    :p2_polar, :p2_size, :p2_flex, :p2_h_doner, :p2_h_acceptor, :p2_pi_doner, :p2_pi_acceptor, :p2_polarisable, :p2_sigma, :p2_branch,
    :p3_polar, :p3_size, :p3_flex, :p3_h_doner, :p3_h_acceptor, :p3_pi_doner, :p3_pi_acceptor, :p3_polarisable, :p3_sigma, :p3_branch,
    :p4_polar, :p4_size, :p4_flex, :p4_h_doner, :p4_h_acceptor, :p4_pi_doner, :p4_pi_acceptor, :p4_polarisable, :p4_sigma, :p4_branch,
    :p5_polar, :p5_size, :p5_flex, :p5_h_doner, :p5_h_acceptor, :p5_pi_doner, :p5_pi_acceptor, :p5_polarisable, :p5_sigma, :p5_branch,
    :p6_polar, :p6_size, :p6_flex, :p6_h_doner, :p6_h_acceptor, :p6_pi_doner, :p6_pi_acceptor, :p6_polarisable, :p6_sigma, :p6_branch,
    :activity,
] :activity
preprocess(::Triazines) = (X -> coerce(X, :p2_pi_doner => Continuous))
# These two columns are all zeros. We could keep them, but they are not
# useful and may confuse some algorithms.
drop_columns(::Triazines) = [:p5_flex, :p5_h_doner]

url(::Triazines) = "https://www.dcc.fc.up.pt/~ltorgo/Regression/triazines.tgz"

################################################################################
# Servo (ID=87) 1986
#-------------------------------------------------------------------------------
# - 167 instances
# - 4 attributes
#-------------------------------------------------------------------------------
# Data was from a simulation of a servo system
#-------------------------------------------------------------------------------
# # Creators
#  - Karl Ulrich ()
################################################################################

@dataset Small Servo datasetdir("servo") [
    :motor, :screw, :pgain, :vgain, :class
] :class
preprocess(::Servo) = (X -> coerce(X, :motor => Multiclass, :screw => Multiclass))

url(::Servo) = "https://archive.ics.uci.edu/dataset/87/servo"
doi(::Servo) = "10.24432/C5Q30F"

################################################################################
# Iris
################################################################################

@dataset CategoricalDataSet Iris datasetdir("iris") [
    :sepal_length, :sepal_width, :petal_length, :petal_width, :species
] :species

url(::Iris) = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

################################################################################
# Wine
################################################################################

@dataset CategoricalDataSet Wine datasetdir("wine") [
    :class, :alcohol, :malic_acid, :ash, :alcalinity_of_ash, :magnesium, :total_phenols, :flavanoids, :nonflavanoid_phenols, :proanthocyanins, :color_intensity, :hue, :od280_od315_of_diluted_wines, :proline
] :class

url(::Wine) = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

################################################################################
# Breast Cancer (Diagnostic)
################################################################################

@dataset CategoricalDataSet CancerDiagnostic datasetdir("cancer_class") [
    :id, :diagnosis,
    :radius_mean, :texture_mean, :perimeter_mean, :area_mean, :smoothness_mean, :compactness_mean, :concavity_mean, :concave_points_mean, :symmetry_mean, :fractal_dimension_mean,
    :radius_se, :texture_se, :perimeter_se, :area_se, :smoothness_se, :compactness_se, :concavity_se, :concave_points_se, :symmetry_se, :fractal_dimension_se,
    :radius_worst, :texture_worst, :perimeter_worst, :area_worst, :smoothness_worst, :compactness_worst, :concavity_worst, :concave_points_worst, :symmetry_worst, :fractal_dimension_worst
] :diagnosis
drop_columns(::CancerDiagnostic) = [:id]

url(::CancerDiagnostic) = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

doi(::Union{Servo,Iris,Wine,CancerDiagnostic}) = "10.1109/TNN.2009.2036259"

################################################################################
# MNIST
################################################################################

abstract type ChoSaul <: CategoricalDataSet end
doi(::ChoSaul) = "10.1162/NECO_a_00018"
struct MNIST <: ChoSaul end

const mnist = MNIST()
push!(all, mnist)

raw_data(::MNIST) = MNISTData(; split=:train), MNISTData(; split=:test)

preprocess(::MNIST) = function (data)
    train, test = data

    X = cat(train.features, test.features; dims=3)
    y = vcat(train.targets, test.targets)

    # Flatten images into vectors
    Xflat = reshape(X, :, size(X, 3))'

    return table(Xflat), categorical(y)
end

@memoize unpack(ds::MNIST) = data(ds)

url(::MNIST) = "http://yann.lecun.com/exdb/mnist/"

################################################################################
# Delve Datasets
#
# The ones we use have different variants with different number of features,
# linearity and noise.

#   size = 8 or 32 (number of features)
#   linearity = "f" or "n" (fairly linear or non-linear)
#   noise = "h" or "m" (high or medium)
#
################################################################################

abstract type Delve <: DataSet end
abstract type DelveRegressionDataSet <: Delve end

is_regression(::DelveRegressionDataSet) = true

url(::Delve) = "http://www.cs.toronto.edu/~delve/data/datasets.html"

name(ds::Delve) = lowercase(split(string(typeof(ds)), ".")[end]) * "-$(ds.size)$(ds.linearity)$(ds.noise)"

function path(ds::Delve) # Most times Dataset.data.gz is available, but not always
    compressed = datasetdir(name(ds), "Dataset.data.gz")
    if isfile(compressed)
        return compressed
    end
    return datasetdir(name(ds), "Dataset.data")
end

function Base.show(io::IO, ds::Delve)
    type_name = string(typeof(ds))
    name = split(type_name, '.') |> last
    print(io, name, ds.size, ds.linearity, ds.noise)
end
Base.show(io::IO, ::MIME"text/plain", ds::Delve) = print(io, typeof(ds), "(", ds.size, ", ", ds.linearity, ", ", ds.noise, ")")

function raw_data(ds::Delve)
    CSV.read(path(ds), DataFrame; header=header(ds),
        delim=" ",
        ignorerepeated=true
    )
end

# The target is the last column
target(ds::Delve) = Symbol("Column$(ds.size+1)")

macro delve_dataset(name, type=Delve)
    it = Iterators.product((32, 8), "fn", "hm")

    definition = esc(quote
        struct $name <: $type
            size::Int
            linearity::Char
            noise::Char
        end
    end)

    declarations = map(it) do (size, linearity, noise)
        esc(quote
            function $(Symbol(name, size, linearity, noise))()
                return $name($size, $linearity, $noise)
            end

            push!(all, $(Symbol(name, size, linearity, noise))())
        end)
    end

    return quote
        $definition
        $(declarations...)
    end
end

@delve_dataset Bank DelveRegressionDataSet
@delve_dataset Pumadyn DelveRegressionDataSet

@dataset CategoricalDataSet Ionosphere datasetdir("ionosphere", "ionosphere.data") false :Column35
url(::Ionosphere) = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
doi(::Ionosphere) = "10.24432/C5W01B"

@dataset CategoricalDataSet Adult datasetdir("adult", "adult.data") [
    :age, :workclass, :fnlwgt, :education, :education_num, :marital_status,
    :occupation, :relationship, :race, :sex, :capital_gain, :capital_loss,
    :hours_per_week, :native_country, :income] :income
url(::Adult) = "https://archive.ics.uci.edu/dataset/2/adult"
doi(::Adult) = "10.24432/C5XW20"

################################################################################
# Synthetic Datasets
################################################################################

abstract type CatSynthetic <: CategoricalDataSet end
abstract type RegSynthetic <: RegressionDataSet end

is_synthetic(::DataSet) = false
is_synthetic(::Union{CatSynthetic,RegSynthetic}) = true

@kwdef struct Blobs <: CatSynthetic
    N::Int = 500
    p::Int = 2
    centers::Int = 3
    cluster_std::Union{Float64,Vector{Float64}} = 1.75
    rng::Int = 1234
end

raw_data(ds::Blobs) = MLJ.make_blobs(ds.N, ds.p; ds.centers, ds.cluster_std, ds.rng)
@memoize unpack(ds::Blobs) = data(ds)
preprocess(::Blobs) = identity
Base.show(io::IO, ds::Blobs) = print(io, typeof(ds), "(", ds.N, ",", ds.p, ",", ds.centers, ",", ds.cluster_std, ",", ds.rng, ")")

@kwdef struct Regression <: RegSynthetic
    N::Int = 300
    p::Int = 5
    noise::Float64 = 0.5
    sparse::Float64 = 0.2
    outliers::Float64 = 0.1
    rng::Int = 1234
end

raw_data(ds::Regression) = MLJ.make_regression(ds.N, ds.p; ds.noise, ds.sparse, ds.outliers, ds.rng)
unpack(ds::Regression) = data(ds)
preprocess(::Regression) = identity
Base.show(io::IO, ds::Regression) = print(io, typeof(ds), "(", ds.N, ",", ds.p, ",", ds.noise, ",", ds.sparse, ",", ds.outliers, ",", ds.rng, ")")

################################################################################


################################################################################
# Statlog (German Credit Data) (144) 1994
#-------------------------------------------------------------------------------
# - 1000 instances
# - 20 attributes
#-------------------------------------------------------------------------------
# This dataset classifies people described by a set of attributes as good or bad credit risks. Comes in two formats (one all numeric). Also comes with a cost matrix
################################################################################
@dataset CategoricalDataSet StatlogGermanCreditData datasetdir("statlog_german_credit_data", "german.data") [
    :Attribute1, :Attribute2, :Attribute3, :Attribute4, :Attribute5, :Attribute6, :Attribute7, :Attribute8, :Attribute9, :Attribute10, :Attribute11, :Attribute12, :Attribute13, :Attribute14, :Attribute15, :Attribute16, :Attribute17, :Attribute18, :Attribute19, :Attribute20, :class,
] :class

preprocess(::StatlogGermanCreditData) = X -> coerce(X,
    :Attribute1 => Multiclass,
    :Attribute2 => Count,
    :Attribute3 => Multiclass,
    :Attribute4 => Multiclass,
    :Attribute5 => Count,
    :Attribute6 => Multiclass,
    :Attribute7 => Multiclass,
    :Attribute8 => Count,
    :Attribute9 => Multiclass,
    :Attribute10 => Multiclass,
    :Attribute11 => Count,
    :Attribute12 => Multiclass,
    :Attribute13 => Count,
    :Attribute14 => Multiclass,
    :Attribute15 => Multiclass,
    :Attribute16 => Count,
    :Attribute17 => Multiclass,
    :Attribute18 => Count,
    :Attribute19 => Finite{2},
    :Attribute20 => Finite{2},
    :class => Finite{2},
)

url(::StatlogGermanCreditData) = "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data"
doi(::StatlogGermanCreditData) = "10.24432/C5NC77"

################################################################################
# Automobile (10) 1985
#-------------------------------------------------------------------------------
# - 205 instances
# - 25 attributes
#-------------------------------------------------------------------------------
# From 1985 Ward's Automotive Yearbook
################################################################################
# @dataset RegressionDataSet Automobile datasetdir("automobile", "imports-85.data") [
#     :symboling, :normalized_losses, :make, :fuel_type, :aspiration, :num_of_doors, :body_style, :drive_wheels,
#     :engine_location, :wheel_base, :length, :width, :height, :curb_weight, :engine_type, :num_of_cylinders, :engine_size,
#     :fuel_system, :bore, :stroke, :compression_ratio, :horsepower, :peak_rpm, :city_mpg, :highway_mpg, :price,
# ] :price
#
# preprocess(::Automobile) = X -> coerce(X,
#     :fuel_system => Multiclass,
#     :num_of_cylinders => Count,
#     :engine_type => Multiclass,
#     :engine_location => Finite{2},
#     :drive_wheels => Multiclass,
#     :body_style => Multiclass,
#     :num_of_doors => Count,
#     :aspiration => Finite{2},
#     :fuel_type => Finite{2},
#     :make => Multiclass,
#     :symboling => Count,
# )
#
# url(::Automobile) = "https://archive.ics.uci.edu/dataset/10/automobile"
# doi(::Automobile) = "10.24432/C5B01C"

################################################################################
# Auto MPG (ID=9) 1993
#-------------------------------------------------------------------------------
# - 398 instances
# - 7 attributes
#-------------------------------------------------------------------------------
# Revised from CMU StatLib library, data concerns city-cycle fuel consumption
#-------------------------------------------------------------------------------
# # Creators
#
#  - R. Quinlan ()
# 
#-------------------------------------------------------------------------------
# # Attribute Information
# 
# - displacement: (Continuous Feature) .
# - mpg: (Continuous Target) .
# - cylinders: (Integer Feature) .
# - horsepower: (Continuous Feature) .
# - weight: (Continuous Feature) .
# - acceleration: (Continuous Feature) .
# - model_year: (Integer Feature) .
# - origin: (Integer Feature) .
# - car_name: (Categorical ID) .
#
################################################################################

@dataset RegressionDataSet AutoMpg datasetdir("auto_mpg", "auto-mpg.data") [
    :Mpg, :Cylinders, :Displacement, :Horsepower, :Weight, :Acceleration, :ModelYear, :Origin, :CarName,
] :Mpg

read_data(ds::AutoMpg; kwargs...) = CSV.read(path(ds), DataFrame; header=header(ds), ignorerepeated=true, delim="  ", missingstring="?", drop=[:CarName], silencewarnings=true, kwargs...)
preprocess(::AutoMpg) = X -> let
    X[!, :Origin] = map(x -> parse(Int, x[1]), X[!, :Origin])
    dropmissing!(X)
    mapcols!(collect, X)
    coerce(X,
        :Cylinders => Count,
        :ModelYear => Count,
        :Origin => Count,
        :CarName => Multiclass,
    )
end

select_columns(::AutoMpg) = [:Displacement, :Cylinders, :Horsepower, :Weight, :Acceleration, :ModelYear, :Origin, :Mpg]

url(::AutoMpg) = "https://archive.ics.uci.edu/dataset/9/auto+mpg"
doi(::AutoMpg) = "10.24432/C5859H"

################################################################################
# Glass Identification (ID=42) 1987
#-------------------------------------------------------------------------------
# - 214 instances
# - 9 attributes
#-------------------------------------------------------------------------------
# From USA Forensic Science Service; 6 types of glass; defined in terms of their oxide content (i.e. Na, Fe, K, etc)
#-------------------------------------------------------------------------------
# # Creators
#
#  - B. German ()
# 
#-------------------------------------------------------------------------------
# # Attribute Information
# 
# - Id_number: (Integer ID) .
# - RI: (Continuous Feature) refractive index.
# - Na: (Continuous Feature) Sodium.
# - Mg: (Continuous Feature) Magnesium.
# - Al: (Continuous Feature) Aluminum.
# - Si: (Continuous Feature) Silicon.
# - K: (Continuous Feature) Potassium.
# - Ca: (Continuous Feature) Calcium.
# - Ba: (Continuous Feature) Barium.
# - Fe: (Continuous Feature) Iron.
# - Type_of_glass: (Categorical Target) .
#
################################################################################

@dataset CategoricalDataSet GlassIdentification datasetdir("glass_identification", "glass.data") [
    :IdNumber, :Ri, :Na, :Mg, :Al, :Si, :K, :Ca, :Ba, :Fe, :TypeOfGlass,
] :TypeOfGlass

preprocess(::GlassIdentification) = X -> coerce(X,
    :IdNumber => Count,
    :TypeOfGlass => Multiclass,
)

drop_colums(::GlassIdentification) = [:IdNumber,]

url(::GlassIdentification) = "https://archive.ics.uci.edu/dataset/42/glass+identification"
doi(::GlassIdentification) = "10.24432/C5WW2P"

################################################################################
# Hepatitis (ID=46) 1983
#-------------------------------------------------------------------------------
# - 155 instances
# - 19 attributes
#-------------------------------------------------------------------------------
# From G.Gong: CMU; Mostly Boolean or numeric-valued attribute types; Includes cost data (donated by Peter Turney)
#-------------------------------------------------------------------------------
# # Creators
#
# 
#-------------------------------------------------------------------------------
# # Attribute Information
# 
# - Class: (Categorical Target) .
# - Age: (Integer Feature) .
# - Sex: (Categorical Feature) .
# - Steroid: (Categorical Feature) .
# - Antivirals: (Categorical Feature) .
# - Fatigue: (Categorical Feature) .
# - Malaise: (Categorical Feature) .
# - Anorexia: (Categorical Feature) .
# - Liver Big: (Categorical Feature) .
# - Liver Firm: (Categorical Feature) .
# - Spleen Palpable: (Categorical Feature) .
# - Spiders: (Categorical Feature) .
# - Ascites: (Categorical Feature) .
# - Varices: (Categorical Feature) .
# - Bilirubin: (Continuous Feature) .
# - Alk Phosphate: (Integer Feature) .
# - Sgot: (Integer Feature) .
# - Albumin: (Integer Feature) .
# - Protime: (Integer Feature) .
# - Histology: (Integer Feature) .
#
################################################################################

@dataset CategoricalDataSet Hepatitis datasetdir("hepatitis", "hepatitis.data") [
    :Class, :Age, :Sex, :Steroid, :Antivirals, :Fatigue, :Malaise, :Anorexia, :LiverBig, :LiverFirm, :SpleenPalpable, :Spiders, :Ascites, :Varices, :Bilirubin, :AlkPhosphate, :Sgot, :Albumin, :Protime, :Histology,
] :Class

read_data(ds::Hepatitis; kwargs...) = CSV.read(path(ds), DataFrame; header=header(ds), missingstring="?", kwargs...)

preprocess(::Hepatitis) = X -> let
    dropmissing!(X)
    mapcols!(collect, X)
    coerce(X,
        :Class => Multiclass,
        :Age => Count,
        :Sex => Multiclass,
        :Steroid => Multiclass,
        :Antivirals => Multiclass,
        :Fatigue => Multiclass,
        :Malaise => Multiclass,
        :Anorexia => Multiclass,
        :LiverBig => Multiclass,
        :LiverFirm => Multiclass,
        :SpleenPalpable => Multiclass,
        :Spiders => Multiclass,
        :Ascites => Multiclass,
        :Varices => Multiclass,
        :AlkPhosphate => Count,
        :Sgot => Count,
        :Protime => Count,
        :Histology => Finite{2},
    )
end

url(::Hepatitis) = "https://archive.ics.uci.edu/dataset/46/hepatitis"
doi(::Hepatitis) = "10.24432/C5Q59J"


################################################################################
# Bike Sharing Dataset (ID=275) 2013
#-------------------------------------------------------------------------------
# - 17389 instances
# - 16 attributes
#-------------------------------------------------------------------------------
# This dataset contains the hourly and daily count of rental bikes between years 2011 and 2012 in Capital bikeshare system with the corresponding weather and seasonal information.
#-------------------------------------------------------------------------------
# # Creators
#
#  - Hadi Fanaee-T ()
# 
#-------------------------------------------------------------------------------
# # Attribute Information
# 
#
################################################################################

abstract type BikeSharingDataset <: RegressionDataSet end

@dataset BikeSharingDataset BikeSharingDatasetDay datasetdir("bike_sharing_dataset/day.csv") 1 :cnt
@dataset BikeSharingDataset BikeSharingDatasetHour datasetdir("bike_sharing_dataset/hour.csv") 1 :cnt

Base.string(::BikeSharingDatasetHour) = "BikeSharingHour"
Base.string(::BikeSharingDatasetDay) = "BikeSharingDay"

url(::BikeSharingDataset) = "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"
doi(::BikeSharingDataset) = "10.24432/C5W894"

################################################################################
# Communities and Crime (ID=183) 2002
#-------------------------------------------------------------------------------
# - 1994 instances
# - 128 attributes
#-------------------------------------------------------------------------------
# Communities within the United States. The data combines socio-economic data from the 1990 US Census, law enforcement data from the 1990 US LEMAS survey, and crime data from the 1995 FBI UCR.
#-------------------------------------------------------------------------------
# # Creators
#
#  - Michael Redmond ()
# 
#-------------------------------------------------------------------------------
# # Attribute Information
# 
#
################################################################################

@dataset RegressionDataSet CommunitiesAndCrime datasetdir("communities_and_crime/communities.data") false :Column128

read_data(ds::CommunitiesAndCrime; kwargs...) = CSV.read(path(ds), DataFrame; header=header(ds), missingstring="?", kwargs...) |> dropmissing

url(::CommunitiesAndCrime) = "https://archive.ics.uci.edu/dataset/183/communities+and+crime"
doi(::CommunitiesAndCrime) = "10.24432/C53W3X"

end
