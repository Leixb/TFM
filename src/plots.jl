"""
# Plots

This module contains the functions used to plot the data obtained in the experiments.

The aim is to have all the figures that will be used in the final document properly generated and
saved in the figures folder in a way that they can be reproduced in the final document.

For the final document, the figures should be saved in vector format (pdf, eps, svg, etc)
with optional pasteurized fragments using CairoMakie.

For other purposed, we can use GLMakie of WGLMakie to quickly visualize and iterate on the plots.
"""
module Plots

using CairoMakie, GLMakie, LaTeXStrings, AlgebraOfGraphics, MathTeXEngine
using DataFrames, DataFramesMeta, MLJ, DrWatson
using Printf, Dates

function tex_theme!()
    Makie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))
end

import ..Utils, ..Experiments, ..DataSets

function plot_asin(interactive=Makie.current_backend() == GLMakie)
    fig = Figure(fonts=(;regular="Latin Modern Roman"))
    ax = Axis(fig[1, 1])
    x = range(-2, 2, length=200)

    if interactive
        sg = SliderGrid(fig[2, 1],
            (label = "σ", range=-6:1:6, format="10^{}", startvalue=0)
        )

        sliderobservables = [s.value for s in sg.sliders]
        values = lift(sliderobservables...) do slvalues...
            Utils.kernel_asin_normalized.(x, 0, 10.0^slvalues[1])
        end

        lines!(ax, x, values)
    else
        sigma_values = range(-3, 3, step=3)

        for sigma in sigma_values
            values = Utils.kernel_asin_normalized.(x, 0, 10.0^sigma)
            lines!(ax, x, values, label=latexstring("10^{$sigma}"))
        end
        axislegend(L"\sigma_w")
    end

    ylims!(ax, 0, 1)
    xlims!(ax, extrema(x))

    fig
end

function plot_kernel_3d_grid(kernel, args...; offset = pi/5, dims=(2, 2), kwargs...)
    fig = Figure()
    xs = range(-2, 2, length=100)
	z = [ kernel(x, y, args...; kwargs...) for x in xs, y in xs]

    axes = [fig[x, y] for y in 1:dims[1], x in 1:dims[2]]
	axes = reshape(axes, 1, :)

	angles = LinRange(0, pi/2, length(axes)) .+ offset

	for (ax, azimuth) in zip(axes, angles)
		ax1 = Axis3(ax; azimuth)
		surface!(ax1, xs, xs, z)
	end

    fig

end

function plot_kernel_3d_interactive(kernel, args...; kwargs...)
    fig = Figure()
    xs = range(-2, 2, length=100)

    ax = Axis3(fig[1, 1])
    sg = SliderGrid(fig[2, 1],
        (label = "σ", range=-6:1:6, format="10^{}", startvalue=0)
    )

    sliderobservables = [s.value for s in sg.sliders]
    values = lift(sliderobservables...) do slvalues...
        [kernel(x, y, 10.0^slvalues[1], args...; kwargs...) for x in xs, y in xs]
    end

    on(sg.sliders[1].value) do value
        autolimits!(ax)
    end

    surface!(ax, xs, xs, values)

    fig

end

plot_kernel_3d(args...; interactive=Makie.current_backend() == GLMakie, kwargs...) =
    if interactive
        plot_kernel_3d_interactive(args...; kwargs...)
    else
        plot_kernel_3d_grid(args...; kwargs...)
    end


function experiment_data(folder="svms", scan=true)
    if scan 
	    df = collect_results!(
            datadir(folder);
            black_list=Experiments.default_ignore_results()
        )
    else
        df = wload(datadir("results_$folder.jld2"))["df"]
    end
	df.kernel_cat = categorical(string.(df.kernel))
	df.dataset_cat = categorical(string.(df.dataset))
    df.sigma = Utils.gamma2sigma.(df.gamma)
	df.kernel_family = map(x -> string(x)[1:4], df.kernel_cat)
	df.cost = round.(df.cost, sigdigits=2)
	df.cost_cat = map(df.cost) do cost @sprintf("%.0E", cost) end
	df.ms = @. Dates.value(df.duration)
	df.ms_per_iter = @. df.ms / df.n_iter / 5
	df
end

function summarize_best(df, grouping::AbstractArray=[:dataset_cat, :kernel_cat], value=:measure_test, maximum=false)
    # If maximum, we reverse the order
    @chain df begin
        sort(value, rev=maximum)
        groupby(grouping)
        combine(first)
    end
end

is_regression(ds) = ds isa DataSets.RegressionDataSet || ds isa DataSets.DelveRegressionDataSet

regression(df) = @rsubset(df, is_regression(:dataset))
classification(df) = @rsubset(df, !is_regression(:dataset))

function plot_best(df)
    cols = mapping(
        :kernel_cat,
        :measure_test=>"nRMSE",
		color=:kernel_cat=>"Kernel"
	)
	grp = mapping(layout = :dataset_cat)
	geom = visual(BarPlot)

	plt = data(df) * cols * grp * geom
	fg = draw(plt, facet = (; linkyaxes = :none), axis=(;xticklabelrotation=pi/4))
end

# function plot_sigma(df, show_kernels=["Asin", "AsinNorm"], linkyaxis=false,
#     show_rbf = true
# )
    # cols = mapping(
    #     :sigma,
    #     :measure_test=>"nRMSE",
    #     color=:kernel_cat=>"Kernel"
    # )
    # grp = mapping(layout = :dataset_cat)
    # geom = visual(ScatterLines)
	# plt = data(df) * cols * grp * geom
	# fg = draw(plt, facet = (; linkyaxes = :none), axis=(;xscale=log10, xticklabelrotation=pi/4))
# end

function plot_sigma(df, show_kernels=["Asin", "AsinNorm"], args...,
    ;linkyaxis=false, linkxaxis=false, show_rbf = false,
    interactive=Makie.current_backend() == GLMakie, kwargs...
)

    fig = Figure(args...; kwargs...)

    datasets = unique(df.dataset_cat)
    kernels = unique(df.kernel_cat)
    sort!(kernels, by = x -> (startswith(x, "Acos") ? "B$x" : "A$x"))

    wcolors = Makie.wong_colors()

    kernel_colors = Dict(k => wcolors[i] for (i, k) in enumerate(kernels))

    toggles_dict = Dict(k =>
        if interactive
            Toggle(fig, active=k in show_kernels, buttoncolor=kernel_colors[k])
        else
            (;active=k in show_kernels)
        end
        for k in kernels)

    if interactive
        toggles_dict["RadialBasis"].active = show_rbf
    end

    gr = GridLayout(fig[1, 1])

    n = Int(ceil(sqrt(length(datasets))))
    m = Int(ceil(length(datasets) / n))

    axes = [Axis(gr[j, i]) for i in 1:m, j in 1:n if (i-1)*n+j <= length(datasets)]
    axes = reshape(axes, 1, :)

    df_groups = @chain df begin
        sort(:sigma)
        groupby(:dataset_cat)
    end

    for (ax, df_group) in zip(axes, df_groups)
        dataset = df_group.dataset_cat[1]

        foreach(groupby(df_group, :kernel_cat)) do df_kernel
            kernel = df_kernel.kernel_cat[1]

            if kernel == "RadialBasis"
                if show_rbf || interactive
                    hline = hlines!(ax, [minimum(df_kernel.measure_test)], color=kernel_colors[kernel], linewidth=2, label="RBF (best)")
                    interactive && connect!(hline.visible, toggles_dict[kernel].active)
                end

                # lines = lines!(ax, df_kernel.sigma, df_kernel.measure_test,
                    # linestyle = :dot,
                    # label=string(kernel), visible = true, color=kernel_colors[kernel])
                # connect!(lines.visible, toggles_dict[kernel].active)
            elseif kernel in show_kernels || interactive

                slines = lines!(ax, df_kernel.sigma, df_kernel.measure_test,
                    linestyle = df_kernel.kernel_family[1] == "Acos" ? :dash : :solid,
                    label=string(kernel), visible = true, color=kernel_colors[kernel])

                spoints = scatter!(ax, df_kernel.sigma, df_kernel.measure_test,
                    label=string(kernel), visible = true, color=kernel_colors[kernel])

                if interactive
                    connect!(slines.visible, toggles_dict[kernel].active)
                    connect!(spoints.visible, toggles_dict[kernel].active)
                end
            end
        end
        ax.title = string(dataset)
        ax.xscale = log10
    end

    Label(fig[1, 0], text = "nRMSE", font = :bold, fontsize = 20, tellheight = false,  rotation = pi/2)
    Label(fig[2, 1], text = L"\sigma_w", font = :bold, fontsize = 20, tellwidth = false)
    Label(fig[0, 1:2], text = "Sigma vs nRMSE by Dataset", font = :bold, fontsize = 20, tellwidth = false)

    linkyaxis && linkyaxes!(axes...)
    linkxaxis && linkxaxes!(axes...)

    if !interactive
        Legend(fig[1, 2], axes[1], "Kernels", merge=true, framevisible=false)
        return fig
    end

    toggles = collect(pairs(toggles_dict))

    sort!(toggles, by = x -> x[1])

    labels = map(toggles) do (kernel, _)
        Label(fig, string(kernel))
    end
    toggles = map(toggles) do (_, toggle) toggle end

    fig[1, 2] = grid!(
        hcat(toggles, labels),
        tellheight = false)

    fig[1, 2][0, 1:2] = Label(fig, "Kernels", font=:bold)

    foreach(toggles) do toggle
        on(toggle.active) do _
            map(autolimits!, axes)
        end
    end

    linkyaxes_toggle = Toggle(fig, active=linkyaxis)
    linkxaxes_toggle = Toggle(fig, active=linkxaxis)

    customize_toggles = [linkyaxes_toggle, linkxaxes_toggle]
    customize_labels = [Label(fig, "Link Y"), Label(fig, "Link X")]

    fig[1, 2][length(toggles)+1, 1:2] = grid!(
        hcat(customize_toggles, customize_labels),
        tellheight = false
    )

    on(linkyaxes_toggle.active) do active
        active ? linkyaxes!(axes...) : unlinkyaxes!(axes...)
    end
    on(linkxaxes_toggle.active) do active
        active ? linkxaxes!(axes...) : unlinkxaxes!(axes...)
    end

    fig

end

"""
Reverse the effects of `linkaxes!` on the given axes.
"""
function unlinkaxes!(dir::Union{Val{:x}, Val{:y}}, a::Axis, others...)
    axes = Axis[a; others...]
    for ax in axes
        setproperty!(ax, dir isa Val{:x} ? :xaxislinks : :yaxislinks, Vector{Axis}())
        reset_limits!(ax)
    end
end

function unlinkaxes!(a::Axis, others...)
    unlinkxaxes!(a, others...)
    unlinkyaxes!(a, others...)
end

unlinkxaxes!(a::Axis, others...) = unlinkaxes!(Val(:x), a, others...)
unlinkyaxes!(a::Axis, others...) = unlinkaxes!(Val(:y), a, others...)

end
