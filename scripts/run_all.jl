#!/usr/bin/env julia

using MLJ
using TFM
using Serialization

using LIBSVM: Kernel

kernels = [
    Kernel.Asin
    Kernel.AsinNorm
    Kernel.RadialBasis
]

machines = Dict()

for ds in DataSets.all
    @info "Running $(ds)..."
    if ds in [DataSets.cancer, DataSets.mnist]
        @warn "Skipping $(ds)"
        continue
    end
    (Xtrain, Xtest), (ytrain, ytest) = partition(ds)
    for kernel in kernels
        @info "Using $(kernel)..."

        filename="$(ds)_$(kernel).jld2"
        if isfile(filename)
            @info "$(filename) already exists, loading and skipping..."
            machines["$(ds) $(kernel)"] = machine(filename)
            continue
        end

        pipe = Models.pipeline(ds; kernel, max_iter=Int32(1e5))
        tuned = Models.tuned_model(pipe; acceleration_resampling=CPUProcesses())
        mach = machine(tuned, Xtrain, ytrain)
        fit!(mach)

        machines["$(ds) $(kernel)"] = mach
        MLJ.save(filename, mach)

        @info report(mach)
    end
end

serialize("machines_new.jls", machines)
