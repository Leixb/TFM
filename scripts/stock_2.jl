### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ df8ef62c-c9c9-11ed-1567-b1e6560ed443
using DrWatson
@quickactivate "tfm"

# ╔═╡ df2cab88-b171-4d0a-9669-1dd7340ed824
begin
    # using Distributed;
    # addprocs(6, exeflags=`--project=..`)

	using CSV
	using DataFrames
	using MLJ
	using DrWatson
	using MLJTuning
	using LIBSVM
	using LinearAlgebra

    LIBSVM.libsvm_set_verbose(true)
    # LIBSVM.set_num_threads(-1)
end

include(srcdir("helpers.jl"))
include(srcdir("datasets.jl"))

df, target = Main.DataSets.stock()
Xtrain, Xtest, ytrain, ytest = prepare_dataset(df, target; ratio=0.9, rng=1234)

# standardize
Xstandardizer = machine(Standardizer(), Xtrain)
MLJ.fit!(Xstandardizer)

Xtrain = MLJ.transform(Xstandardizer, Xtrain)
Xtest = MLJ.transform(Xstandardizer, Xtest)

# yStandardizer = machine(Standardizer(), ytrain)
# MLJ.fit!(yStandardizer)

# ytrain = MLJ.transform(yStandardizer, ytrain)
# ytest = MLJ.transform(yStandardizer, ytest)


# ╔═╡ 78e7b8f8-97d7-4215-8d65-94dfc1dde717
EpsilonSVR = @load EpsilonSVR pkg=LIBSVM

# ╔═╡ 58268741-c3a9-47b6-8fd5-f245a1b10621
kernel = LIBSVM.Kernel.RadialBasis
# kernel = LIBSVM.Kernel.AsinNorm
# kernel = LIBSVM.Kernel.Asin
# kernel = LIBSVM.Kernel.Acos0

# ╔═╡ f1791ea4-8eb3-4c15-b02a-a8e952b987c3
model = EpsilonSVR(kernel=kernel)

# ╔═╡ 673d6b8c-32f9-48bd-bb3b-c046aa0470f2
gamma_values = let
    sigma_range = 10.0 .^ (-3:1.0:3)
    sigma2gamma = sigma -> 1 ./ (2 .*sigma.^2)
    sigma2gamma(sigma_range)
end

# ╔═╡ e4a80537-61fc-4f3c-9582-c6bcae4718d6
hyper_grid = [
	# range(model, :gamma, values = gamma_values), # Asin
    # range(model, :gamma, values = 10 .^ (-3:1.0:0)), # RBF
    range(model, :gamma, values = 10 .^ (-3:1.0:3)), # RBF
  	range(model, :epsilon, values = 10 .^ (-5:1.0:1)),
	range(model, :cost, values = 10 .^ (-2:1.0:6))
]

# ╔═╡ 2b031d1e-c981-4ef5-b39e-de7b099249b3
tuned_model = TunedModel(model=model,
                        resampling = CV(nfolds=5),
                        tuning = Grid(resolution=10),
                        range = hyper_grid,
                        measure = rms,
                        # acceleration_resampling = CPUProcesses(),
);

# ╔═╡ 89d8db63-2675-41ae-9b64-7491af9f0917
# Skip model and vendor for the moment since its not clear what they did in the paper
pipe = # FeatureSelector(features=[:Model, :Vendor], ignore=true) |>
	# OneHotEncoder() |>
	# ContinuousEncoder() |>
    Standardizer() |>
	tuned_model

# ╔═╡ 5c211fce-e3a8-4b83-8fe2-76b219bea81f
mach = machine(pipe, Xtrain, ytrain)

# ╔═╡ 4991646e-da69-454c-af5a-a4e3aa0618f0
MLJ.fit!(mach)

# ╔═╡ e82d87fe-d5a3-4b7d-9937-b39662cd5fa3
yhat = MLJ.predict(mach, Xtest)

# ╔═╡ b5963d0a-5f59-4e65-8389-9ac8a272a3f3
begin
    println(yhat)
    println(ytest)
    println(mean((ytest .- yhat).^2))
end

# ╔═╡ 1c19b1fe-6668-4893-86e2-37d51ee5a439
MLJ.save("model_stock2.jls", mach)




# ╔═╡ Cell order:
# ╠═df8ef62c-c9c9-11ed-1567-b1e6560ed443
# ╠═df2cab88-b171-4d0a-9669-1dd7340ed824
# ╠═d91bf710-af14-425b-9157-e5bafe9db65a
# ╠═cb85b9f6-691e-4b3e-8617-b835768b47be
# ╠═c434fc35-a93d-4fce-8dbd-8cb57238d9b8
# ╠═0ad6cd41-2790-40d6-bb3b-9048a32268e4
# ╠═78e7b8f8-97d7-4215-8d65-94dfc1dde717
# ╠═58268741-c3a9-47b6-8fd5-f245a1b10621
# ╠═f1791ea4-8eb3-4c15-b02a-a8e952b987c3
# ╠═673d6b8c-32f9-48bd-bb3b-c046aa0470f2
# ╠═e4a80537-61fc-4f3c-9582-c6bcae4718d6
# ╠═2b031d1e-c981-4ef5-b39e-de7b099249b3
# ╠═89d8db63-2675-41ae-9b64-7491af9f0917
# ╠═5c211fce-e3a8-4b83-8fe2-76b219bea81f
# ╠═4991646e-da69-454c-af5a-a4e3aa0618f0
# ╠═e82d87fe-d5a3-4b7d-9937-b39662cd5fa3
# ╠═b5963d0a-5f59-4e65-8389-9ac8a272a3f3
# ╠═1c19b1fe-6668-4893-86e2-37d51ee5a439
