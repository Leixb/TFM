using MLJ
using LIBSVM: Kernel

using TFM: DataSets, Models

EpsilonSVR = @load EpsilonSVR pkg=LIBSVM

df, target = DataSets.stock()

display(schema(df))

y, X = unpack(df, ==(target); shuffle=false)

(Xtrain, Xtest), (ytrain, ytest) = partition((X, y), 0.8, rng=1234, multi=true)

kernel = Kernel.RadialBasis

regressor = EpsilonSVR(kernel=kernel)

gamma_values = if kernel == Kernel.RadialBasis
        10 .^ (-3.0:1.0:0)
    else let
        sigma_range = 10.0 .^ (-3:1.0:3)
        sigma2gamma = sigma -> 1 ./ (2 .*sigma.^2)
        sigma2gamma(sigma_range)
    end
end

hyper_grid = [
    # range(regressor, :gamma, values = gamma_values),
    # range(regressor, :epsilon, values = 10 .^ (-5:1.0:1)),
    # range(regressor, :cost, values = 10 .^ (-2:1.0:6))
    range(regressor, :cost, values = 10 .^ (-1:1.0:0))
]

display(hyper_grid)

tunedRegressor = TunedModel(
    model=regressor,
    tuning=Grid(goal=100),
    resampling=CV(nfolds=5),
    measure=rms,
    range=hyper_grid,
    acceleration_resampling=CPUProcesses()
)

model = ContinuousEncoder() |>
    Standardizer() |>
    TransformedTargetModel(
        # regressor;
        tunedRegressor;
        transformer=Standardizer()
    )

mach = machine(model, Xtrain, ytrain)

fit!(mach)

yhat = predict(mach, Xtest)

println("kernel: ", kernel)
println("mse: ", rms(yhat, ytest)^2)
println("yhat: ", yhat[1:10])
println("ytest: ", ytest[1:10])

# MLJ.save("model_tuning.jls", mach)
