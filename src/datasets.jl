#!/usr/bin/env julia

module DataSets

import DrWatson.datadir
import CSV
import DataFrames.DataFrame
using MLJ: coerce, Multiclass, Continuous

# DataSet is an abstract type that represents a dataset.

abstract type DataSet end

data(ds::DataSet) = raw_data(ds) |> preprocess(ds)
raw_data(ds::DataSet) = CSV.read(path(ds), DataFrame; header=header(ds))

header(ds::DataSet) = false
preprocess(::DataSet) = identity

# These methods must be implemented for each DataSet type.
# Optionally, you can implement header(ds::DataSet) to specify
# the column names of the dataset, and preprocess(ds::DataSet)
# to transform the raw data into a form suitable for MLJ.

target(ds::DataSet) = error("target not implemented for $(typeof(ds))")
path(ds::DataSet) = error("path not implemented for $(typeof(ds))")

# DataSets that are used in Frenay and Verleysen (2016)
abstract type FrenayDataSet <: DataSet end
abstract type LargeDataSet <: FrenayDataSet end
abstract type SmallDataSet <: FrenayDataSet end

struct Abalone <: LargeDataSet end
struct Ailerons <: LargeDataSet end
struct CompActs <: LargeDataSet end
struct Elevators <: LargeDataSet end

struct Cancer <: SmallDataSet end
struct CPU <: SmallDataSet end
struct Stock <: SmallDataSet end
struct Triazines <: SmallDataSet end

abalone = Abalone()
ailerons = Ailerons()
cancer = Cancer()
compacts = CompActs()
cpu = CPU()
elevators = Elevators()
stock = Stock()
triazines = Triazines()

all_datasets = [
    abalone,
    ailerons,
    cancer,
    compacts,
    cpu,
    elevators,
    stock,
    triazines,
]


# https://archive.ics.uci.edu/ml/datasets/Abalone

path(::Abalone) = datadir("exp_raw", "abalone")
header(::Abalone) = [:Sex, :Length, :Diameter, :Height, :Whole_weight, :Shucked_weight, :Viscera_weight, :Shell_weight, :Rings]
target(::Abalone) = :Rings
preprocess(ds::Abalone) = X -> coerce(X, target(ds) => Continuous)


# https://archive.ics.uci.edu/ml/datasets/Ailerons

path(::Ailerons) = datadir("exp_raw", "ailerons", "ailerons.data")
header(::Ailerons) = [
    :climbRate, :Sgz, :p, :q, :curPitch, :curRoll, :absRoll, :diffClb, :diffRollRate, :diffDiffClb,
    :SeTime1, :SeTime2, :SeTime3, :SeTime4, :SeTime5, :SeTime6, :SeTime7,
    :SeTime8, :SeTime9, :SeTime10, :SeTime11, :SeTime12, :SeTime13, :SeTime14,
    :diffSeTime1, :diffSeTime2, :diffSeTime3, :diffSeTime4, :diffSeTime5, :diffSeTime6, :diffSeTime7,
    :diffSeTime8, :diffSeTime9, :diffSeTime10, :diffSeTime11, :diffSeTime12, :diffSeTime13, :diffSeTime14,
    :alpha, :Se, :goal,
]
target(::Ailerons) = :goal


# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

path(::Cancer) = datadir("exp_raw", "cancer")
target(::Cancer) = :Column2

# https://archive.ics.uci.edu/ml/datasets/Computer+Hardware

path(::CompActs) = datadir("exp_raw", "compActs", "cpu_act.data")
header(::CompActs) = [
    :lread, :lwrite, :scall, :sread, :swrite, :fork, :exec, :rchar, :wchar,
    :pgout, :ppgout, :pgfree, :pgscan, :atch, :pgin, :ppgin, :pflt, :vflt,
    :runqsz, :freemem, :freeswap, :usr
]
target(::CompActs) = :usr


# https://archive.ics.uci.edu/ml/datasets/Computer+Hardware

path(::CPU) = datadir("exp_raw", "cpu")
header(::CPU) = [:Vendor, :Model, :MYCT, :MMIN, :MMAX, :CACH, :CHMIN, :CHMAX, :PRP, :ERP]
target(::CPU) = :ERP
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

# Elevators

path(::Elevators) = datadir("exp_raw", "elevators", "elevators.data")
header(::Elevators) = [
    :climbRate, :Sgz, :p, :q, :curRoll, :absRoll, :diffClb, :diffRollRate, :diffDiffClb,
    :SaTime1, :SaTime2, :SaTime3, :SaTime4, :diffSaTime1, :diffSaTime2, :diffSaTime3, :diffSaTime4,
    :Sa, :Goal
]
target(::Elevators) = :Goal

# Stock

path(::Stock) = datadir("exp_raw", "stock", "stock.data")
header(::Stock) = [
    :Company1, :Company2, :Company3, :Company4, :Company5,
    :Company6, :Company7, :Company8, :Company9, :Company10,
]
target(::Stock) = :Company10
function raw_data(ds::Stock)
    CSV.read(path(ds), DataFrame; header=header(ds),
        delim="\t",
        ignorerepeated=true
    )
end

# Triazines

path(::Triazines) = datadir("exp_raw", "triazines", "triazines.data")
header(::Triazines) = [
    :p1_polar, :p1_size, :p1_flex, :p1_h_doner, :p1_h_acceptor, :p1_pi_doner, :p1_pi_acceptor, :p1_polarisable, :p1_sigma, :p1_branch,
    :p2_polar, :p2_size, :p2_flex, :p2_h_doner, :p2_h_acceptor, :p2_pi_doner, :p2_pi_acceptor, :p2_polarisable, :p2_sigma, :p2_branch,
    :p3_polar, :p3_size, :p3_flex, :p3_h_doner, :p3_h_acceptor, :p3_pi_doner, :p3_pi_acceptor, :p3_polarisable, :p3_sigma, :p3_branch,
    :p4_polar, :p4_size, :p4_flex, :p4_h_doner, :p4_h_acceptor, :p4_pi_doner, :p4_pi_acceptor, :p4_polarisable, :p4_sigma, :p4_branch,
    :p5_polar, :p5_size, :p5_flex, :p5_h_doner, :p5_h_acceptor, :p5_pi_doner, :p5_pi_acceptor, :p5_polarisable, :p5_sigma, :p5_branch,
    :p6_polar, :p6_size, :p6_flex, :p6_h_doner, :p6_h_acceptor, :p6_pi_doner, :p6_pi_acceptor, :p6_polarisable, :p6_sigma, :p6_branch,
    :activity,
]
target(::Triazines) = :activity
preprocess(::Triazines) = (X -> coerce(X, :p2_pi_doner => Continuous))

end
