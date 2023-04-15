using DrWatson
@quickactivate "tfm"

using MLJ

# Here you may include files from the source directory
include(srcdir("helpers.jl"))
include(srcdir("datasets.jl"))

println(
    """
    Currently active project is: $(projectname())
    Path of active project: $(projectdir())
    """
)

df, target = Main.DataSets.stock()

using MLJ
using LIBSVM.Kernel

Xtrain, Xtest, ytrain, ytest = prepare_dataset(df, target; ratio=0.9, rng=423789)

EpsilonSVR = @load EpsilonSVR pkg=LIBSVM

kernel = Kernel.RadialBasis
# kernel = Kernel.AsinNorm
# kernel = Kernel.Asin
# kernel = Kernel.Acos0

display(schema(df))

model = EpsilonSVR(kernel=kernel)

gamma_values = let
    sigma_range = 10.0 .^ (-3:1.0:3)
    sigma2gamma = sigma -> 1 ./ (2 .*sigma.^2)
    sigma2gamma(sigma_range)
end

hyper_grid = [
    range(model, :gamma, values = gamma_values),
    range(model, :epsilon, values = 10 .^ (-5:1.0:1)),
    range(model, :cost, values = 10 .^ (-2:1.0:6))
]

tuned_model = TunedModel(model=model,
                        resampling = CV(nfolds=5),
                        tuning = Grid(),
                        range = hyper_grid,
                        measure = rms,
                        # acceleration_resampling = CPUProcesses(),
);


pipe = ContinuousEncoder() |>
	# OneHotEncoder() |>
    Standardizer() |>
	tuned_model
pipe =
    (X -> coerce(X, target=>Continuous)) |>
    TransformedTargetModel(pipe; transformer=Standardizer())

mach = machine(pipe, Xtrain, ytrain)

mach_fitted = MLJ.fit!(mach)

yhat = MLJ.predict(mach, Xtest)

println(mean((ytest .- yhat).^2))

MLJ.save("stock_asin.jls", mach)
