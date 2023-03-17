### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 4d68a848-c2b6-11ed-09eb-8d1cf2b01ecb
begin
	using Pkg;
	Pkg.activate("..")
end

# ╔═╡ 9247f9ee-1245-4481-93cb-42684a006d57
begin
	using CSV
	using DataFrames
	using DrWatson
	using Printf
	using MLJ
	using MLJTuning
	using LIBSVM
	using LinearAlgebra
end

# ╔═╡ 36517ba9-8b90-4b9a-86a6-78d221da8f67
df = coerce(CSV.read(datadir("exp_raw", "abalone"), DataFrame;
	header=[:Sex, :Length, :Diameter, :Height, :Whole_weight,
			:Shucked_weight, :Viscera_weight, :Shell_weight, :Rings]
), :Rings => Continuous)

# ╔═╡ 59cfbc5f-687b-4e2d-aae7-1bf368eee789
schema(df)

# ╔═╡ 3d543c30-177b-464b-80ab-e4c260140334
y, X = unpack(df, ==(:Rings); shuffle=false)

# ╔═╡ 5925166e-7ac7-4b80-a095-be10dbdaff96
(Xtrain, Xval), (ytrain, yval) = partition((X, y), 0.9, rng=1234, multi=true)

# ╔═╡ 01761688-43b4-4a0b-b05b-5e48a6722c4e
EpsilonSVR = @load EpsilonSVR pkg=LIBSVM

# ╔═╡ 904d554d-e176-4d86-a9f6-764eab712b03
kernel = #LIBSVM.Kernel.RadialBasis
	(x, y) -> exp(-norm(x - y)^2 / 2)

# ╔═╡ 1e909542-3606-48cb-8357-5d2f114e6957


# ╔═╡ 10fcf690-c27c-40de-93b2-ddcd8627cbc1
model = EpsilonSVR(kernel=kernel)

# ╔═╡ 4dd4501c-0ae3-42e9-8dae-508c17bb417d
hyper_grid = [
	range(model, :gamma, values = [0.01, 0.1, 1.0, 10.0, 100.0]),
  	range(model, :epsilon, values = [0.01, 0.1, 1.0, 10.0, 100.0]),
]

# ╔═╡ dcc30d45-db7a-4d75-9c25-b55d1e4c01d6
tuned_model = TunedModel(model=model,
                        resampling = CV(nfolds=3),
                        tuning = Grid(resolution=10),
                        range = hyper_grid,
                        measure = rms);

# ╔═╡ a746e396-d698-4a37-8310-3216194a0fe6
pipe = (X -> coerce(X, :Sex=>Multiclass, :Rings => Continuous)) |>
	# ContinuousEncoder() |>
	OneHotEncoder() |>
	tuned_model

# ╔═╡ f024abbe-6fde-4b68-af36-8e3a079104a5
mach = machine(pipe, Xtrain, ytrain)

# ╔═╡ c4a59d88-2422-434f-b384-a0fe5eb1d211
mach_fitted = MLJ.fit!(mach)

# ╔═╡ 108e3e48-bcc7-4ad1-926c-ce0d5002f3d3
fitted_params(mach).deterministic_tuned_model.best_fitted_params

# ╔═╡ b2fe674a-fe08-42e4-80e9-1fe3e4efb8e8
yval_pred = MLJ.predict(mach, Xval)

# ╔═╡ 7d0cef96-55de-4b5c-a427-5923a36a61ce
rms(yval, yval_pred)

# ╔═╡ eaa7f627-612b-4c31-8eba-8f163b9e9ad2
root_mean_squared_error(yval, yval_pred)^2

# ╔═╡ e268f993-088b-4c19-aedc-40fe64a49229
doc("EpsilonSVR", pkg="LIBSVM")

# ╔═╡ Cell order:
# ╠═4d68a848-c2b6-11ed-09eb-8d1cf2b01ecb
# ╠═9247f9ee-1245-4481-93cb-42684a006d57
# ╠═36517ba9-8b90-4b9a-86a6-78d221da8f67
# ╠═59cfbc5f-687b-4e2d-aae7-1bf368eee789
# ╠═3d543c30-177b-464b-80ab-e4c260140334
# ╠═5925166e-7ac7-4b80-a095-be10dbdaff96
# ╠═01761688-43b4-4a0b-b05b-5e48a6722c4e
# ╠═904d554d-e176-4d86-a9f6-764eab712b03
# ╠═1e909542-3606-48cb-8357-5d2f114e6957
# ╠═10fcf690-c27c-40de-93b2-ddcd8627cbc1
# ╠═4dd4501c-0ae3-42e9-8dae-508c17bb417d
# ╠═dcc30d45-db7a-4d75-9c25-b55d1e4c01d6
# ╠═a746e396-d698-4a37-8310-3216194a0fe6
# ╠═f024abbe-6fde-4b68-af36-8e3a079104a5
# ╠═c4a59d88-2422-434f-b384-a0fe5eb1d211
# ╠═108e3e48-bcc7-4ad1-926c-ce0d5002f3d3
# ╠═b2fe674a-fe08-42e4-80e9-1fe3e4efb8e8
# ╠═7d0cef96-55de-4b5c-a427-5923a36a61ce
# ╠═eaa7f627-612b-4c31-8eba-8f163b9e9ad2
# ╠═e268f993-088b-4c19-aedc-40fe64a49229
