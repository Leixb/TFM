#!/usr/bin/env julia

"""
Module to specify all datasets used in the project.

This should contain the methods necessary to read and do minimal
pre-processing. For models, see `models.jl`.
"""
module DataSets

import CSV
import DataFrames.DataFrame
using LIBSVM: Kernel

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
function unpack(ds::DataSet; args...)
    if ds.X === nothing || ds.y === nothing || args != ()
        ds.X, ds.y = unpack(data(ds), !=(target(ds)); args...)
    end
    ds.X, ds.y
end

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
doi(ds::DataSet) = error("doi not implemented for $(typeof(ds))")

"List of all datasets defined in the module"
all = []

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
        @kwdef mutable struct $name <: $type
            X::Union{DataFrame,Nothing} = nothing
            y::Union{Vector{Real},Nothing} = nothing
        end
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

abstract type Frenay <: RegressionDataSet end
doi(::Frenay) = "10.1016/j.neucom.2010.11.037"

# DataSet relative size according to Frenay and Verleysen (2016)
abstract type Large <: Frenay end
abstract type Small <: Frenay end

datasetdir(path...) = joinpath(ENV["DATASETS"], path...)

################################################################################
# Abalone
################################################################################

@dataset Large Abalone datasetdir("abalone") [
    :Sex, :Length, :Diameter, :Height, :Whole_weight, :Shucked_weight, :Viscera_weight, :Shell_weight, :Rings
] :Rings
preprocess(ds::Abalone) = X -> coerce(X, target(ds) => Continuous, :Sex => Multiclass)

url(::Abalone) = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"

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
select_columns(::Ailerons) = [:climbRate, :Sgz, :p, :q, :curPitch, :curRoll, :absRoll]

url(::Ailerons) = "https://www.dcc.fc.up.pt/~ltorgo/Regression/ailerons.tgz"

################################################################################
# Wisconsin Breast Cancer (Prognostic)
################################################################################

@dataset Small Cancer datasetdir("cancer") false :Column3
# Drop ID and diagnosis
drop_columns(::Cancer) = [:Column1, :Column2]

url(::Cancer) = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

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
select_columns(::Elevators) = [:climbRate, :Sgz, :p, :q, :curRoll, :absRoll]

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
# MNIST
################################################################################

import Tables

abstract type ChoSaul <: CategoricalDataSet end
doi(::ChoSaul) = "10.1162/NECO_a_00018"
@kwdef mutable struct MNIST <: ChoSaul
    X::Union{Nothing,Tables.MatrixTable} = nothing
    y::Union{Nothing,Vector{UInt8}} = nothing
end

const mnist = MNIST()
push!(all, mnist)

raw_data(::MNIST) = MNISTData(;split=:train), MNISTData(;split=:test)

preprocess(::MNIST) = function(data)
    train, test = data

    X = cat(train.features, test.features; dims=3)
    y = vcat(train.targets, test.targets)

    # Flatten images into vectors
    Xflat = reshape(X, :, size(X, 3))'

    return table(Xflat), categorical(y)
end

function unpack(ds::MNIST)
    if ds.X === nothing || ds.y === nothing
        ds.X, ds.y = data(ds)
    end
    ds.X, ds.y
end

url(::MNIST) = "http://yann.lecun.com/exdb/mnist/"

end
