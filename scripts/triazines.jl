### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

using DrWatson
@quickactivate "tfm"

# ╔═╡ df2cab88-b171-4d0a-9669-1dd7340ed824
begin
    using Distributed;
    addprocs(6, exeflags=`--project=..`)

	using CSV
	using DataFrames
	using MLJ
	using DrWatson
	using MLJTuning
	using LIBSVM
	using LinearAlgebra

    LIBSVM.libsvm_set_verbose(true)
    LIBSVM.set_num_threads(-1)
end

# ╔═╡ d91bf710-af14-425b-9157-e5bafe9db65a
df = CSV.read(datadir("exp_raw", "triazines", "triazines.data"), DataFrame;
    header=[
        :p1_polar, :p1_size, :p1_flex, :p1_h_doner, :p1_h_acceptor, :p1_pi_doner, :p1_pi_acceptor, :p1_polarisable, :p1_sigma, :p1_branch,
        :p2_polar, :p2_size, :p2_flex, :p2_h_doner, :p2_h_acceptor, :p2_pi_doner, :p2_pi_acceptor, :p2_polarisable, :p2_sigma, :p2_branch,
        :p3_polar, :p3_size, :p3_flex, :p3_h_doner, :p3_h_acceptor, :p3_pi_doner, :p3_pi_acceptor, :p3_polarisable, :p3_sigma, :p3_branch,
        :p4_polar, :p4_size, :p4_flex, :p4_h_doner, :p4_h_acceptor, :p4_pi_doner, :p4_pi_acceptor, :p4_polarisable, :p4_sigma, :p4_branch,
        :p5_polar, :p5_size, :p5_flex, :p5_h_doner, :p5_h_acceptor, :p5_pi_doner, :p5_pi_acceptor, :p5_polarisable, :p5_sigma, :p5_branch,
        :p6_polar, :p6_size, :p6_flex, :p6_h_doner, :p6_h_acceptor, :p6_pi_doner, :p6_pi_acceptor, :p6_polarisable, :p6_sigma, :p6_branch,
        :activity,
    ]
) |> (X -> 
	coerce(X, :p2_pi_doner => Continuous))

# ╔═╡ cb85b9f6-691e-4b3e-8617-b835768b47be
schema(df)

# ╔═╡ c434fc35-a93d-4fce-8dbd-8cb57238d9b8
y, X = unpack(df, ==(:activity); shuffle=false)

# ╔═╡ 0ad6cd41-2790-40d6-bb3b-9048a32268e4
(Xtrain, Xtest), (ytrain, ytest) = partition((X, y), 0.9, rng=1234, multi=true)

# ╔═╡ 78e7b8f8-97d7-4215-8d65-94dfc1dde717
EpsilonSVR = @load EpsilonSVR pkg=LIBSVM

# ╔═╡ 58268741-c3a9-47b6-8fd5-f245a1b10621
# kernel = LIBSVM.Kernel.RadialBasis
kernel = LIBSVM.Kernel.Acos0
# kernel = LIBSVM.Kernel.Asin
# kernel = LIBSVM.Kernel.AsinNorm

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
	range(model, :gamma, values = gamma_values),
  	range(model, :epsilon, values = 10 .^ (-5:1.0:1)),
	range(model, :cost, values = 10 .^ (-2:1.0:6))
]

# ╔═╡ 2b031d1e-c981-4ef5-b39e-de7b099249b3
tuned_model = TunedModel(model=model,
                        resampling = CV(nfolds=5),
                        tuning = Grid(resolution=10),
                        range = hyper_grid,
                        measure = rms,
                        acceleration_resampling = CPUProcesses(),
                        # acceleration = CPUProcesses(),
);

# ╔═╡ 89d8db63-2675-41ae-9b64-7491af9f0917
pipe = 
	# OneHotEncoder() |>
	ContinuousEncoder() |>
    # Standardizer() |>
	tuned_model

# ╔═╡ 5c211fce-e3a8-4b83-8fe2-76b219bea81f
mach = machine(pipe, Xtrain, ytrain)

# ╔═╡ 4991646e-da69-454c-af5a-a4e3aa0618f0
mach_fitted = MLJ.fit!(mach)

# ╔═╡ e82d87fe-d5a3-4b7d-9937-b39662cd5fa3
ytest_pred = MLJ.predict(mach, Xtest)

# ╔═╡ b5963d0a-5f59-4e65-8389-9ac8a272a3f3
println(root_mean_squared_error(ytest, ytest_pred)^2)

# ╔═╡ 1c19b1fe-6668-4893-86e2-37d51ee5a439
MLJ.save("model_tri.jls", mach)

# ╔═╡ b2ec4972-f380-456f-9516-4aa61d6fa784
mean((ytest - ytest_pred).^2)

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
# ╠═b2ec4972-f380-456f-9516-4aa61d6fa784
