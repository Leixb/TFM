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

using MLJ
import MLJ: unpack, partition

using MLDatasets: MNIST as MNISTData

"""
DataSet is the top abstract type that provides the interface for reading
and preprocessing a dataset.
"""
abstract type DataSet end

# These methods should only be implemented if the dataset is not
# in a proper CSV format.
data(ds::DataSet) = raw_data(ds) |> preprocess(ds)
raw_data(ds::DataSet) = CSV.read(path(ds), DataFrame; header=header(ds))

# These methods should be implemented for each dataset
target(ds::DataSet) = error("target not implemented for $(typeof(ds))")
path(ds::DataSet) = error("path not implemented for $(typeof(ds))")

# The following methods are optional, but highly recommended
header(ds::DataSet) = false
preprocess(::DataSet) = identity

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

abstract type Frenay <: CategoricalDataSet end
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
preprocess(ds::Abalone) = X -> coerce(X, target(ds) => Continuous)

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

url(::Ailerons) = "https://www.dcc.fc.up.pt/~ltorgo/Regression/ailerons.tgz"

################################################################################
# Wisconsin Breast Cancer
################################################################################

@dataset Small Cancer datasetdir("cancer") false :Column2

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
preprocess(::CPU) = (X -> coerce(X,
    :Model => Multiclass,
    :Vendor => Multiclass,
    :MYCT => Continuous,
    :MMIN => Continuous,
    :MMAX => Continuous,
    :CACH => Continuous,
    :CHMIN => Continuous,
    :CHMAX => Continuous,
    :PRP => Continuous,
    :ERP => Continuous
))

url(::CPU) = "https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data"

################################################################################
# Elevators
################################################################################

@dataset Large Elevators datasetdir("elevators", "elevators.data") [
    :climbRate, :Sgz, :p, :q, :curRoll, :absRoll, :diffClb, :diffRollRate, :diffDiffClb,
    :SaTime1, :SaTime2, :SaTime3, :SaTime4, :diffSaTime1, :diffSaTime2, :diffSaTime3, :diffSaTime4,
    :Sa, :Goal
] :Goal

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

url(::Triazines) = "https://www.dcc.fc.up.pt/~ltorgo/Regression/triazines.tgz"

################################################################################
# MNIST
################################################################################

abstract type ChoSaul <: CategoricalDataSet end
doi(::ChoSaul) = "10.1162/NECO_a_00018"
struct MNIST <: ChoSaul end

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

unpack(ds::MNIST) = data(ds)

url(::MNIST) = "http://yann.lecun.com/exdb/mnist/"

end
