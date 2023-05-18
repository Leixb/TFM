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

using Makie, LaTeXStrings, AlgebraOfGraphics, MathTeXEngine
using DataFrames, DataFramesMeta, MLJ, DrWatson
using Printf, Dates

is_interactive() = string(Makie.current_backend()) == "GLMakie"

function tex_theme!()
    Makie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))
end

import ..Utils, ..Experiments, ..DataSets
import ..DataSets: is_regression
import ..Measures

function plot_asin(interactive=is_interactive())
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
		surface!(ax1, xs, xs, z, rasterize=true)
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

plot_kernel_3d(args...; interactive=is_interactive(), kwargs...) =
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
        combine(first, nrow)
    end
end

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
    dims=nothing, show_bands=false,
    interactive=is_interactive(), kwargs...
)

    fig = Figure(args...; kwargs...)

    datasets = unique(df.dataset_cat)
    kernels = unique(df.kernel_cat)

    # force order: asin..., acos..., rbf
    sort!(kernels, by = x -> (startswith(x, "Asin") ? "A$x" : "B$x"))
    wcolors = Makie.wong_colors()

    kernel_colors = Dict(k => wcolors[i] for (i, k) in enumerate(kernels))

    # We only add std bands toggle if they are requested
    # initially since they may not be valid
    if interactive && show_bands
        bands_toggle = Toggle(fig, active=true)
    end

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

    if !(dims isa Nothing)
        (m, n) = dims
    else
        n = Int(ceil(sqrt(length(datasets))))
        m = Int(ceil(length(datasets) / n))
    end

    axes = [Axis(gr[i, j]) for j in 1:m, i in 1:n if (i-1)*n+j <= length(datasets)]

    df_groups = @chain df begin
        sort(:sigma)
        groupby(:dataset_cat)
    end

    for (ax, df_group) in zip(axes, df_groups)
        dataset = df_group.dataset_cat[1]
        df_subgroup = groupby(df_group, :kernel_cat)

        # First we plot bands so they are on bottom
        if show_bands
            for df_kernel in df_subgroup
                kernel = df_kernel.kernel_cat[1]
                if !(kernel in show_kernels)
                    continue
                end

                # WARN: measure_test and std are not really related
                # this assumes that we override measure_test to the
                # proper column when doing show_bands...
                bands = band!(
                    ax,
                    df_kernel.sigma,
                    df_kernel.measure_test .- df_kernel.std,
                    df_kernel.measure_test .+ df_kernel.std,
                    label=string(kernel), visible=true,
                    color=(kernel_colors[kernel], 0.3)
                )

                if interactive
                    connect!(bands.visible, bands_toggle.active)
                end
            end
        end

        for df_kernel in df_subgroup
            kernel = df_kernel.kernel_cat[1]

            if kernel == "RadialBasis"
                if show_rbf || interactive
                    hline = hlines!(ax, [minimum(df_kernel.measure_test)], color=kernel_colors[kernel], linewidth=2, label="RBF (best)", linestyle=:dash)
                    interactive && connect!(hline.visible, toggles_dict[kernel].active)
                end

                # lines = lines!(ax, df_kernel.sigma, df_kernel.measure_test,
                    # linestyle = :dot,
                    # label=string(kernel), visible = true, color=kernel_colors[kernel])
                # connect!(lines.visible, toggles_dict[kernel].active)
                continue
            elseif !(kernel in show_kernels || interactive)
                continue
            end

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
        ax.title = string(dataset)
        ax.xscale = log10
    end

    # Get string representation of measure and resampling for the labels
    measure = df.measure[1]
    measure_name = if measure isa Measures.MSE
        "MSE"
    elseif measure isa Measures.nRMSE
        "nRMSE"
    else
        # Fallback to io.show with titlecase applied
        titlecase(string(measure))
    end

    resampling = string(df.resampling[1])

    Label(fig[1, 0], text = measure_name, font = :bold, fontsize = 20, tellheight = false,  rotation = pi/2)
    Label(fig[2, 1], text = L"\sigma_w", font = :bold, fontsize = 20, tellwidth = false)
    Label(fig[0, 1:2], text = "Sigma vs $measure_name by Dataset ($resampling)", font = :bold, fontsize = 20, tellwidth = false)

    if linkxaxis && !interactive
        linkxaxes!(axes...)
        for ax in axes[1:end-m]
            hidexdecorations!(ax; grid=false, minorgrid=false)
        end
    end

    if linkyaxis && !interactive
        linkyaxes!(axes...)
        for i in (setdiff(Set(1:length(axes)) , Set(1:m:length(axes))))
            hideydecorations!(axes[i]; grid=false, minorgrid=false)
        end
    end

    trim!(gr)

    if !interactive
        Legend(fig[1, 2], axes[1], "Kernels", merge=true, framevisible=false)
        return fig
    end

    toggles = collect(pairs(toggles_dict))

    sort!(toggles, by = first)

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
    if show_bands
        push!(customize_toggles, bands_toggle)
        push!(customize_labels, Label(fig, "std"))
    end

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

function exec_time(df::DataFrame, show_kernels=["Asin", "AsinNorm"];
	grp = mapping(layout = :cost=>nonnumeric),
	geom = visual(BoxPlot),
    logscale=false, linkyaxes=false)
	cols = mapping(
		:kernel_cat=>"Kernel",
		:n_iter=>"Iterations",
		color=:kernel_cat=>"Kernel",
	)

	plt = data(@chain df begin
		@rtransform(:n_iter=:n_iter+1)
		@rsubset(:kernel_cat in show_kernels)
	end) * cols * geom * grp
	fg = draw(plt,
		facet = (; linkyaxes),
		axis=(;
			xticklabelrotation=pi/4,
			#limits=(nothing, (10, nothing)),
			yscale=logscale ? log10 : identity,
		),
	)
    plt, fg
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

macro saveplot(name, args...)
    if name isa Expr
        args = [name.args[2] ; args...]
        name = name.args[1]
    end

    str_name = string(name) * ".pdf"
    esc(quote
        @info("Plotting " * $str_name)
        $name = $(args...)
        save(plotsdir($str_name), $name)
    end)
end

export @saveplot

end
