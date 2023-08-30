module Benchmark

import Base: length, iterate

using StructTypes
using JSON3
using LIBSVM.Kernel: KERNEL
using Tables
using DataFrames

"""
# Result of a Benchmark execution using `hyperfine` utility.

This is used to parse the JSON output of `hyperfine` and convert it to a
julia `DataFrame`.

Note that these benchmarks are for code outside of `julia`. In our
case, we use it for executions of `libsvm` binaries for comparison
between optimizations.

If possible, using `BenchmarkTools` is preferred.
"""
struct Result
    command::String
    mean::Float64
    stddev::Float64
    median::Float64

    user::Float64
    system::Float64

    min::Float64
    max::Float64

    times::Vector{Float64}
    exit_codes::Vector{UInt8}

    parameters::Dict{String,Any}

    function Result(command::String, mean::Float64, stddev::Float64, median::Float64,
        user::Float64, system::Float64, min::Float64, max::Float64,
        times::Vector{Float64}, exit_codes::Vector{UInt8},
        parameters::Dict{String,Any}
    )
        # If we have a kernel, parameter, we convert it to LIBSVM.Kernel
        if haskey(parameters, "kernel")
            parameters["kernel"] = KERNEL(parse(Int, parameters["kernel"]))
        end
        new(command, mean, stddev, median, user, system, min, max, times, exit_codes, parameters)
    end
end

StructTypes.StructType(::Type{Result}) = StructTypes.Struct()

"""
Needed for JSON3 to parse the file properly, but we only want the inner vector,
so we use Results
"""
@kwdef mutable struct __Results
    results::Vector{Result} = []
end

StructTypes.StructType(::Type{__Results}) = StructTypes.Mutable()

function __Results(filename::String)
    data = read(filename, String)
    JSON3.read(data, __Results)
end

function Results(filename::String)
    res = __Results(filename).results
    @info typeof(res)
    res |> DataFrame
end

# Tables.jl interface

Tables.istable(::Vector{Result}) = true
Tables.rowaccess(::Vector{Result}) = true
Tables.rows(res::Vector{Result}) = res

# We incorporate the inner parameters into the schema
# so that we can access them as columns
function Tables.schema(res::Vector{Result})
    names, types = fieldnames(Result), fieldtypes(Result)

    if length(res) > 0
        # remove parameters from the schema
        names = names[1:end-1]
        types = types[1:end-1]

        names = [names...; Symbol.(keys(res[1].parameters))]
        types = [types...; typeof.(values(res[1].parameters))]
    end
    Tables.Schema(names, types)
end

function Tables.getcolumn(row::Result, col::Symbol)
    if hasproperty(row, col) && col != :parameters
        getproperty(row, col)
    else
        row.parameters[string(col)]
    end
end

function Tables.getcolumn(row::Result, i::Int)
    width = length(fieldnames(Result)) - 1 # -1 for parameters
    if i <= width
        getfield(row, i)
    else
        dict_keys = row.parameters |> keys |> collect |> sort
        row.parameters[dict_keys[i-width]]
    end
end

length(row::Result) = length(fieldnames(Result)) + length(row.parameters)
iterate(row::Result, i::Int=1) = i > length(row) ? nothing : (Tables.getcolumn(row, i), i + 1)

end
