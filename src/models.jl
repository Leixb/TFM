module Models

using MLJ
using LIBSVM: Kernel

EpsilonSVR = @load EpsilonSVR pkg=LIBSVM

export build_model

function build_model(kernel)
    return ContinuousEncoder() |>
        Standardizer() |>
        TransformedTargetModel(EpsilonSVR(kernel=kernel); transformer=Standardizer())
end

function MeanRootSquared(y, ŷ)
    return mean((y .- ŷ).^2)
end

end
