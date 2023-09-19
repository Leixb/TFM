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

# NOTE: We convert the backend to a string to avoid loading GLMakie just to check the backend
is_interactive() = string(Makie.current_backend()) == "GLMakie"

"Makie theme with LaTeX fonts"
function tex_theme!()
    Makie.update_theme!(fonts=(regular=texfont(), bold=texfont(:bold), italic=texfont(:italic)))
end

function no_color_cycle!()
    Makie.update_theme!(
        ScatterLines=(cycle=Cycle([:color, :marker, :linestyle], covary=true),),
        Lines=(cycle=Cycle([:color, :linestyle], covary=true),),
        Scatter=(cycle=Cycle([:color, :marker], covary=true),),
    )
end

import ..Utils, ..Experiments, ..DataSets
import ..DataSets: is_regression
import ..Measures

using DrWatson: projectdir

"Plot the kernel function around the origin with different values of σ"
function plot_kernel(kernel=Utils.kernel_asin_normalized, args...;
    fig_opts::NamedTuple=(),
    interactive=is_interactive(), x=range(-2, 2, length=200), y=0, kwargs...)
    fig = Figure(fonts=(; regular="Latin Modern Roman"); fig_opts...)
    ax = Axis(fig[1, 1])

    if interactive
        sg = SliderGrid(fig[2, 1],
            (label="σ", range=-6:1:6, format="10^{}", startvalue=0)
        )

        sliderobservables = [s.value for s in sg.sliders]
        values = lift(sliderobservables...) do slvalues...
            kernel.(x, y, 10.0^slvalues[1], args...; kwargs...)
        end

        lines!(ax, x, values)
    else

        with_theme(
            Theme(
                Lines=(cycle=Cycle([:color, :linestyle], covary=true),)
            )) do
            sigma_values = range(-3, 3, step=3)

            for sigma in sigma_values
                values = kernel.(x, y, 10.0^sigma, args...; kwargs...)
                lines!(ax, x, values, label=latexstring("10^{$sigma}"))
            end
        end
        # axislegend(L"\sigma_w")
        fig[1, 2] = Legend(fig, ax, L"\sigma_w", framevisible=false)
    end

    ylims!(ax, 0.5, 1)
    xlims!(ax, extrema(x))

    fig
end

function plot_kernel_3d_grid(kernel, args...; offset=pi / 5, dims=(2, 2), kwargs...)
    fig = Figure()
    xs = range(-2, 2, length=100)
    z = [kernel(x, y, args...; kwargs...) for x in xs, y in xs]

    axes = [fig[x, y] for y in 1:dims[1], x in 1:dims[2]]
    axes = reshape(axes, 1, :)

    angles = LinRange(0, pi / 2, length(axes)) .+ offset

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
        (label="σ", range=-6:1:6, format="10^{}", startvalue=0)
    )

    sliderobservables = [s.value for s in sg.sliders]
    values = lift(sliderobservables...) do slvalues...
        [kernel(x, y, 10.0^slvalues[1], args...; kwargs...) for x in xs, y in xs]
    end

    on(sg.sliders[1].value) do _
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


function experiment_data(folder="svms", scan=true; kwargs...)
    if scan
        df = collect_results!(
            datadir(folder);
            black_list=Experiments.SVMExperiment.default_ignore_results(),
            kwargs...
        )
    else
        df = wload(datadir("results_$folder.jld2"))["df"]
    end
    df.kernel_cat = categorical(string.(df.kernel))
    df.dataset_cat = categorical(string.(df.dataset))
    df.sigma = Utils.gamma2sigma.(df.gamma)
    df.kernel_family = map(x -> string(x)[1:4], df.kernel_cat)
    df.cost = round.(df.cost, sigdigits=2)
    df.cost_cat = map(df.cost) do cost
        @sprintf("%.0E", cost)
    end
    df.measure_cv = df.measurement
    df.ms = @. Dates.value(df.duration)
    df.ms_per_iter = @. df.ms / df.n_iter / 5
    @rsubset(df, !(:dataset isa DataSets.Servo))
end

function summarize_best(df, grouping::AbstractArray=[:dataset_cat, :kernel_cat]; by=:measurement, maximum=false)
    # If maximum, we reverse the order
    @chain df begin
        sort(by, rev=maximum)
        groupby(grouping)
        combine(first, nrow)
    end
end

regression(df) = @rsubset(df, is_regression(:dataset))
classification(df) = @rsubset(df, !is_regression(:dataset))

function plot_best(df)
    cols = mapping(
        :kernel_cat,
        :measure_test => "nRMSE",
        color=:kernel_cat => "Kernel"
    )
    grp = mapping(layout=:dataset_cat)
    geom = visual(BarPlot)

    plt = data(df) * cols * grp * geom
    fg = draw(plt, facet=(; linkyaxes=:none), axis=(; xticklabelrotation=pi / 4))
end

function plot_delve(df, dataset::Type{<:DataSets.Delve}, size=32,
    show_kernels=["Asin", "AsinNorm"],
    ; fig=Figure(), linkxaxes=true, linkyaxes=false, show_rbf=false,
    sigma=:sigma, measure=:measure_test, std=:std, show_bands=(measure == :measure_cv),
    interactive=is_interactive()
)
    df = @chain df begin
        @rsubset(:dataset isa dataset)
        @rsubset!(:dataset.size == size)
        sort(:sigma)
    end

    # linearity: fn     fairly / non         X
    # noise: mh         moderate / hight     Y

    # fm   nm
    # fh   nh

    ax = Dict{String,Axis}(
        "fm" => Axis(fig[1, 1], xscale=log10),
        "nm" => Axis(fig[1, 2], xscale=log10, yaxisposition=:right),
        "fh" => Axis(fig[2, 1], xscale=log10),
        "nh" => Axis(fig[2, 2], xscale=log10, yaxisposition=:right),
    )

    kernel_num = Dict(k => i for (i, k) in enumerate(show_kernels))

    do_plot = (name) -> begin
        df_sub = @rsubset(df, :dataset.linearity == name[1], :dataset.noise == name[2])

        if show_bands
            for kernel in show_kernels
                df_sub_kern = @rsubset(df_sub, :kernel_cat == kernel)

                val_sigma = getproperty(df_sub_kern, sigma)
                val_lower = getproperty(df_sub_kern, measure) .- getproperty(df_sub_kern, std)
                val_upper = getproperty(df_sub_kern, measure) .+ getproperty(df_sub_kern, std)

                bands = band!(
                    ax[name],
                    val_sigma, val_lower, val_upper,
                    label=kernel, visible=true,
                    color=(Cycled(kernel_num[kernel]), 0.3)
                )
            end
        end

        for kernel in show_kernels
            df_sub_kern = @rsubset(df_sub, :kernel_cat == kernel)

            scatterlines!(ax[name], getproperty(df_sub_kern, sigma), getproperty(df_sub_kern, measure),
                color=Cycled(kernel_num[kernel]),
                marker=Cycled(kernel_num[kernel]),
                linestyle=Cycled(kernel_num[kernel]),
                label=kernel
            )
        end

        if !show_rbf
            return
        end
        df_rbf = @rsubset(df_sub, :kernel_cat == "RadialBasis")
        if !isempty(df_rbf)
            hlines!(ax[name], [minimum(getproperty(df_rbf, measure))], color=:red, linewidth=2, label="RBF (best)", linestyle=:dot)
        end

        # text!(ax[name], 1, 1, text=name)
    end

    with_theme(
        ScatterLines=(cycle=Cycle([:color, :marker, :linestyle], covary=true),)
    ) do
        map(do_plot, collect(keys(ax)))
    end

    if linkyaxes
        linkyaxes!(values(ax)...)
        hideydecorations!(ax["fm"], grid=false)
        hideydecorations!(ax["fh"], grid=false)
    end

    if linkxaxes
        hidexdecorations!(ax["fm"], grid=false)
        hidexdecorations!(ax["nm"], grid=false)
        linkxaxes!(values(ax)...)
    end

    Box(fig[1, 0])
    Label(fig[1, 0], "Moderate Noise", rotation=pi / 2, tellheight=false)
    Box(fig[2, 0])
    Label(fig[2, 0], "High Noise", rotation=pi / 2, tellheight=false)

    Box(fig[1:2, -1])
    Label(fig[1:2, -1], "Noise Level", rotation=pi / 2)

    Box(fig[0, 1])
    Label(fig[0, 1], "Fairly linear", tellwidth=false)
    Box(fig[0, 2])
    Label(fig[0, 2], "Non-linear", tellwidth=false)

    Box(fig[-1, 1:2])
    Label(fig[-1, 1:2], "Linearity")

    datasetname = split(string(dataset), '.') |> last

    Label(fig[-1:0, -1:0], "$datasetname\n$size", font=:bold, fontsize=20)

    measure = df.measure[1]
    measure_name = if measure isa Measures.MSE
        "MSE"
    elseif measure isa Measures.nRMSE
        "nRMSE"
    else
        # Fallback to io.show with titlecase applied
        titlecase(string(measure))
    end

    Label(fig[1:2, 3], measure_name, rotation=pi / 2, font=:bold)
    Legend(fig[1:2, 4], ax["fm"], "Kernel", framevisible=false, merge=true)
    Label(fig[3, 1:2], L"\sigma_w", font=:bold)

    fig
end

function plot_sigma_subsample(df, show_kernels=["Asin", "AsinNorm"]; linkyaxes=true,
    show_rbf=true, measure=:measure_test
)
    cols = mapping(
        :sigma_scaled,
        measure => "nRMSE",
        color=:subsample => nonnumeric => "Subsample",
        marker=:subsample => nonnumeric => "Subsample",
    )
    grp = mapping(layout=:kernel_cat => "Kernel")
    geom = visual(Scatter)
    plt = data(df) * cols * grp * geom
    fg = draw(plt, facet=(; linkyaxes), axis=(; xscale=log10, xticklabelrotation=pi / 4))

    fg
end

function plot_sigma(df, show_kernels=["Asin", "AsinNorm"], args...,
    ; linkyaxes=false, linkxaxes=false, show_rbf=false,
    dims=nothing, show_bands=false, sigma=:sigma, measure=:measure_test, std=:std,
    interactive=is_interactive(), kwargs...
)

    fig = Figure(args...; kwargs...)

    # Remove kernels that we won't show, so that we don't run into
    # issues with empty plots.
    df_filtered = filter(df) do row
        row.kernel_cat in show_kernels
    end

    datasets = unique(df_filtered.dataset_cat)
    kernels = unique(df.kernel_cat)

    # HACK: this is awful, but it works.
    # We need this to have the datasets with the kernels we want but, without
    # ones which only have radial basis.
    df = filter(df) do row
        row.dataset_cat in datasets
    end

    # force order: asin..., acos..., rbf
    sort!(kernels, by=x -> (startswith(x, "Asin") ? "A$x" : "B$x"))
    wcolors = Makie.wong_colors()

    kernel_idx = Dict(k => i for (i, k) in enumerate(kernels))
    kernel_colors = Dict(k => wcolors[kernel_idx[k]] for k in kernels)

    # We only add std bands toggle if they are requested
    # initially since they may not be valid
    if interactive && show_bands
        bands_toggle = Toggle(fig, active=true)
    end

    toggles_dict = Dict(k =>
        if interactive
            Toggle(fig, active=k in show_kernels, buttoncolor=Cycled(kernel_idx[k]))
        else
            (; active=k in show_kernels)
        end
                        for k in kernels)

    if interactive && haskey(toggles_dict, "RadialBasis")
        toggles_dict["RadialBasis"].active = show_rbf
    end

    gr = GridLayout(fig[1, 1])

    if !(dims isa Nothing)
        (m, n) = dims
    else
        n = Int(ceil(sqrt(length(datasets))))
        m = Int(ceil(length(datasets) / n))
    end

    axes = [Axis(gr[i, j]) for j in 1:m, i in 1:n if (i - 1) * m + j <= length(datasets)]

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

                val_sigma = getproperty(df_kernel, sigma)
                val_lower = getproperty(df_kernel, measure) .- getproperty(df_kernel, std)
                val_upper = getproperty(df_kernel, measure) .+ getproperty(df_kernel, std)

                bands = band!(
                    ax,
                    val_sigma, val_lower, val_upper,
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

            val_sigma = getproperty(df_kernel, sigma)
            val_measure = getproperty(df_kernel, measure)

            if kernel == "RadialBasis"
                if show_rbf || interactive
                    hline = hlines!(ax, [minimum(val_measure)], color=:red, linewidth=2, label="RBF (best)", linestyle=:dot)
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

            slines = scatterlines!(ax, val_sigma, val_measure,
                linestyle=df_kernel.kernel_family[1] == "Acos" ? :dash : :solid,
                marker=Cycled(kernel_idx[kernel]),
                label=string(kernel), visible=true, color=Cycled(kernel_idx[kernel]),
            )

            if interactive
                connect!(slines.visible, toggles_dict[kernel].active)
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

    Label(fig[1, 0], text=measure_name, font=:bold, fontsize=20, tellheight=false, rotation=pi / 2)
    Label(fig[2, 1], text=L"\sigma_w", font=:bold, fontsize=20, tellwidth=false)
    Label(fig[0, 1:2], text="Sigma vs $measure_name by Dataset ($resampling)", font=:bold, fontsize=20, tellwidth=false)

    if linkxaxes && !interactive
        linkxaxes!(axes...)
        for ax in axes[1:end-m]
            hidexdecorations!(ax; grid=false, minorgrid=false)
        end
    end

    if linkyaxes && !interactive
        linkyaxes!(axes...)
        for i in (setdiff(Set(1:length(axes)), Set(1:m:length(axes))))
            hideydecorations!(axes[i]; grid=false, minorgrid=false)
        end
    end

    trim!(gr)

    if !interactive
        Legend(fig[1, 2], axes[1], "Kernels", merge=true, framevisible=false)
        return fig
    end

    toggles = collect(pairs(toggles_dict))

    sort!(toggles, by=first)

    labels = map(toggles) do (kernel, _)
        Label(fig, string(kernel))
    end
    toggles = map(toggles) do (_, toggle)
        toggle
    end

    fig[1, 2] = grid!(
        hcat(toggles, labels),
        tellheight=false)

    fig[1, 2][0, 1:2] = Label(fig, "Kernels", font=:bold)

    foreach(toggles) do toggle
        on(toggle.active) do _
            map(autolimits!, axes)
        end
    end

    linkyaxes_toggle = Toggle(fig, active=linkyaxes)
    linkxaxes_toggle = Toggle(fig, active=linkxaxes)

    customize_toggles = [linkyaxes_toggle, linkxaxes_toggle]
    customize_labels = [Label(fig, "Link Y"), Label(fig, "Link X")]
    if show_bands
        push!(customize_toggles, bands_toggle)
        push!(customize_labels, Label(fig, "std"))
    end

    fig[1, 2][length(toggles)+1, 1:2] = grid!(
        hcat(customize_toggles, customize_labels),
        tellheight=false
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
    grp=mapping(layout=:cost => nonnumeric),
    geom=visual(BoxPlot),
    logscale=false, linkyaxes=false)
    cols = mapping(
        :kernel_cat => "Kernel",
        :n_iter => "Iterations",
        color=:kernel_cat => "Kernel",
    )

    plt = data(@chain df begin
              @rtransform(:n_iter = :n_iter + 1)
              @rsubset(:kernel_cat in show_kernels)
          end) * cols * geom * grp
    fg = draw(plt,
        facet=(; linkyaxes),
        axis=(;
            xticklabelrotation=pi / 4,
            #limits=(nothing, (10, nothing)),
            yscale=logscale ? log10 : identity
        ),
    )
    plt, fg
end

"""
Reverse the effects of `linkaxes!` on the given axes.
"""
function unlinkaxes!(dir::Union{Val{:x},Val{:y}}, a::Axis, others...)
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

"""
Sets the CreationDate of the pdf metadata to zero, so that the file
does not change when the plot is saved again.

WARNING: It is important that timestamp is a string of length 20, otherwise
the pdf metadata will be corrupted.

WARNING: This modified the file in-place, use with caution.
"""
__strip_metadata(filename::String, timestamp="00000000000000+00'00") = run(`sed -i "s/CreationDate.*$/CreationDate (D:$timestamp)/" $filename`)

plotsdocdir(args...) = projectdir("document", "figures", "plots", args...)

macro saveplot(name, args...)
    if name isa Expr
        args = [name.args[2]; args...]
        name = name.args[1]
    end

    str_name = string(name) * ".pdf"

    declaration = esc(quote
        $name = $(args...)
    end)

    saving = esc(quote
        @info("Saving " * $str_name)
        save($plotsdocdir($str_name), $name)
        $__strip_metadata($plotsdocdir($str_name))
    end)

    if isempty(args)
        return esc(quote
            $saving
        end)
    end

    quote
        $declaration
        $saving
    end
end

export @saveplot

end
