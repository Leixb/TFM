#!/usr/bin/env julia

# This script is used to make sure that the PyCall.jl package which is
# used will use the same python environment as the one used by the
# python interpreter.
#
# All this is relevant only when using the custom nix derivation for the
# development environment which sets JULIA_PYCALL_DEPS.

if !haskey(ENV, "JULIA_PYCALL_DEPS")
    println(stderr, "JULIA_PYCALL_DEPS not set, leaving deps.jl as is")
    exit(1)
end

using PyCall

deps = joinpath(dirname(pathof(PyCall)), "..", "deps", "deps.jl")

rm(deps, force=true)
symlink(ENV["JULIA_PYCALL_DEPS"], deps)
