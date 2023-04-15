using MLJ
using LIBSVM: Kernel

using TFM: DataSets, Models

df, target = DataSets.stock()

display(schema(df))

y, X = unpack(df, ==(target); shuffle=false)

(Xtrain, Xtest), (ytrain, ytest) = partition((X, y), 0.8, rng=1234, multi=true)

possible_kernels = [
    Kernel.Linear,
    Kernel.Polynomial,
    Kernel.RadialBasis,
    Kernel.Sigmoid,
    Kernel.Asin,
    Kernel.AsinNorm,
    Kernel.Acos0,
    Kernel.Acos1,
    Kernel.Acos2,
]

models = map(possible_kernels) do kernel
    Models.build_model(kernel)
end

machines = map(models) do model
    machine(model, Xtrain, ytrain)
end

map(fit!, machines)

yhat = map(machines) do mach
    predict(mach, Xtest)
end

for (kernel, yhat) in zip(possible_kernels, yhat)
    println("kernel: ", kernel)
    println("mse: ", rms(yhat, ytest)^2)
    println("yhat: ", yhat[1:10])
end
println("ytest: ", ytest[1:10])
