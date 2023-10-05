
import ..Measures
import ..DataSets

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

    fg
end

function plot_rbf(ax::Axis, df_rbf::AbstractDataFrame, measure::Symbol, measurement_type::Type{<:MLJBase.Measure}; show_std=false)
    # Make sure all rows are "RadialBasis"
    @assert allequal(df_rbf.kernel)
    @assert df_rbf.kernel[1] |> string == "RadialBasis"

    select!(df_rbf, Not(:nrow))
    best = summarize_best(df_rbf, [:kernel_cat], by=measure, maximum=is_maximize(measurement_type))

    hlines!(ax, getproperty(best, measure), label="RBF (best)", color=options.rbf_color, linestyle=ifelse(show_std, :dash, options.rbf_linestyle), linewidth=options.rbf_linewidth)

    if show_std
        hlines!(ax, getproperty(best, measure) .+ best.std, color=options.rbf_color, linestyle=options.rbf_linestyle, linewidth=options.rbf_linewidth)
        hlines!(ax, getproperty(best, measure) .- best.std, color=options.rbf_color, linestyle=options.rbf_linestyle, linewidth=options.rbf_linewidth)
    end
end


function plot_sigma_band(ax, df_sub_kern, kernel; sigma::Symbol, measure::Symbol, std::Symbol=:std)
    val_sigma = getproperty(df_sub_kern, sigma)
    val_lower = getproperty(df_sub_kern, measure) .- getproperty(df_sub_kern, std)
    val_upper = getproperty(df_sub_kern, measure) .+ getproperty(df_sub_kern, std)

    band!(
        ax,
        val_sigma, val_lower, val_upper,
        label=kernel |> string, visible=true,
        color=(kernel_color(kernel), options.band_opacity)
    )
end

function plot_sigma_kernel(ax, df_sub_kern, kernel; sigma::Symbol, measure::Symbol)
    scatterlines!(
        ax,
        getproperty(df_sub_kern, sigma),
        getproperty(df_sub_kern, measure),
        color=kernel_color(kernel),
        marker=Cycled(kernel_idx(kernel)),
        linestyle=Cycled(kernel_idx(kernel)),
        label=kernel |> string,
    )
end

function plot_delve(df::DataFrame, dataset::Type{<:DataSets.Delve},
    size::Integer=32,
    show_kernels::AbstractArray{String}=["Asin", "AsinNorm"],
    ; # Keyword arguments
    fig::Figure=Figure(),
    linkxaxes::Bool=true,
    linkyaxes::Bool=false,
    show_rbf::Bool=false,
    sigma::Symbol=:sigma,
    measure::Symbol=:measure_test,
    std::Symbol=:std,
    show_bands::Bool=(measure == :measure_cv),
    ax_opts::NamedTuple=(; xscale=log10)
)
    # Filter the correct rows and sort by sigma so lines are drawn in order
    df = @chain df begin
        @rsubset(:dataset isa dataset && :dataset.size == size)
        sort(:sigma)
    end

    measure_type = get_measure_type(df)

    # Grid layout 2x2:
    #
    # fm   nm
    # fh   nh
    #
    # linearity: fn     fairly / non         X
    # noise: mh         moderate / hight     Y
    ax = (;
        fm=Axis(fig[1, 1]; ax_opts...),
        fh=Axis(fig[2, 1]; ax_opts...),
        nm=Axis(fig[1, 2]; ax_opts..., yaxisposition=:right),
        nh=Axis(fig[2, 2]; ax_opts..., yaxisposition=:right)
    )

    plot_band = wrap_tuple((ax, df_sub_kern, kernel) -> plot_sigma_band(ax, df_sub_kern, kernel; sigma, measure, std))
    plot_kernel = wrap_tuple((ax, df_sub_kern, kernel) -> plot_sigma_kernel(ax, df_sub_kern, kernel; sigma, measure))

    # List of axes and their corresponding dataframes
    subplots = map(collect(pairs(ax))) do (name, ax)
        name = string(name)
        df_sub = @rsubset(df, :dataset.linearity == name[1], :dataset.noise == name[2])

        ax, df_sub
    end

    # List of axes and their corresponding dataframes by kernel
    plots_and_kernels = map(Iterators.product(subplots, show_kernels)) do ((ax, df_sub), kernel)
        df_sub_kern = @rsubset(df_sub, :kernel_cat == kernel)

        ax, df_sub_kern, kernel
    end

    # Call all the plotting functions
    # We plot bands first so they are on bottom
    show_bands && foreach(plot_band, plots_and_kernels)

    foreach(plot_kernel, plots_and_kernels)

    show_rbf && foreach(subplots) do (ax, df_sub)
        df_rbf = @rsubset(df_sub, :kernel_cat == "RadialBasis")
        if !isempty(df_rbf)
            plot_rbf(ax, df_rbf, measure, measure_type)
        end
    end

    if linkyaxes
        linkyaxes!(values(ax)...)
        hideydecorations!(ax.fm, grid=false)
        hideydecorations!(ax.fh, grid=false)
    end

    if linkxaxes
        hidexdecorations!(ax.fm, grid=false)
        hidexdecorations!(ax.nm, grid=false)
        linkxaxes!(values(ax)...)
    end

    # Facet labels
    BoxLabel(fig[1, 0], "Moderate Noise", rotation=pi / 2, tellheight=false)
    BoxLabel(fig[2, 0], "High Noise", rotation=pi / 2, tellheight=false)
    BoxLabel(fig[1:2, -1], "Noise Level", rotation=pi / 2)

    BoxLabel(fig[0, 1], "Fairly linear", tellwidth=false)
    BoxLabel(fig[0, 2], "Non-linear", tellwidth=false)
    BoxLabel(fig[-1, 1:2], "Linearity")

    # Dataset name and size on top left
    datasetname = name(dataset)
    Label(fig[-1:0, -1:0], "$datasetname\n$size", font=:bold, fontsize=20)

    # Axis labels
    Label(fig[1:2, 3], name(measure_type), rotation=pi / 2, font=:bold)
    Legend(fig[1:2, 4], ax.fm, "Kernel", framevisible=options.framevisible, merge=true)
    Label(fig[3, 1:2], L"\sigma_w", font=:bold)

    fig
end

function plot_sigma_subsample(df, show_kernels=["Asin", "AsinNorm"]; linkyaxes=true,
    show_rbf=true, measure=:measure_test
)
    cols = mapping(
        :sigma_scaled,
        measure => "nRMSE",
        lower=(measure, :std) => -,
        upper=(measure, :std) => +,
        color=:subsample => nonnumeric => "Subsample",
        marker=:subsample => nonnumeric => "Subsample",
    )
    grp = mapping(layout=:kernel_cat => "Kernel")
    geom = visual(LinesFill)
    plt = data(df) * cols * grp * geom
    fg = draw(plt, facet=(; linkyaxes), axis=(; xscale=log10, xticklabelrotation=pi / 4))

    fg
end

function build_grid(pos::GridPosition, len::Int, dims::Union{Nothing,Tuple{Int,Int}}=nothing; kwargs...)
    gr = GridLayout(pos)

    if isnothing(dims)
        n = Int(ceil(sqrt(len)))
        m = Int(ceil(len / n))
    else
        @assert len <= prod(dims)
        (m, n) = dims
    end

    axes = [Axis(gr[i, j]; kwargs...) for j in 1:m, i in 1:n if (i - 1) * m + j <= len]

    return gr, axes, (m, n)
end

function plot_sigma(
    df::DataFrame,
    show_kernels::AbstractArray{String}=["Asin", "AsinNorm"],
    args...,
    ; # Keyword arguments
    linkxaxes::Bool=true,
    linkyaxes::Bool=false,
    show_rbf::Bool=false,
    dims::Union{Nothing,Tuple{Int,Int}}=nothing,
    sigma::Symbol=:sigma,
    measure::Symbol=:measure_test,
    std::Symbol=:std,
    show_bands::Bool=(measure == :measure_cv),
    interactive::Bool=is_interactive(),
    ax_opts::NamedTuple=(; xscale=log10),
    vertical::Bool=false,
    kwargs...
)

    fig = Figure(args...; kwargs...)

    # Remove kernels that we won't show, so that we don't run into
    # issues with empty plots.
    df_filtered = filter(df) do row
        row.kernel_cat in show_kernels
    end

    measure_type = get_measure_type(df_filtered)

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

    # We only add std bands toggle if they are requested
    # initially since they may not be valid
    if interactive && show_bands
        bands_toggle = Toggle(fig, active=true)
    end

    toggles_dict = Dict(k =>
        if interactive
            Toggle(fig, active=k in show_kernels, buttoncolor=Cycled(kernel_color(k)))
        else
            (; active=k in show_kernels)
        end
                        for k in kernels)

    if interactive && haskey(toggles_dict, "RadialBasis")
        toggles_dict["RadialBasis"].active = show_rbf
    end

    gr, axes, (m, n) = build_grid(fig[1, 1], length(datasets), dims; ax_opts...)

    df_groups = @chain df begin
        @orderby(:is_delve, :dataset_cat, :sigma)
        groupby(:dataset_cat, sort=false)
        zip(axes)
    end

    plots_kernels = Iterators.flatmap(df_groups) do (df_group, ax)
        zip(Iterators.repeated(ax), groupby(df_group, :kernel_cat))
    end

    plots_kernels = map(plots_kernels) do (ax, df_subgroup)
        ax, df_subgroup, df_subgroup.kernel_cat[1]
    end

    plots_kernels = filter(plots_kernels) do (_, _, kernel)
        kernel in show_kernels
    end

    plot_band = wrap_tuple((ax, df_sub_kern, kernel) -> plot_sigma_band(ax, df_sub_kern, kernel; sigma, measure, std))
    plot_kernel = wrap_tuple((ax, df_sub_kern, kernel) -> plot_sigma_kernel(ax, df_sub_kern, kernel; sigma, measure))

    if show_bands
        bands = map(plot_band, plots_kernels)
    end
    slines = map(plot_kernel, plots_kernels)

    foreach(df_groups) do (df_group, ax)
        dataset = df_group.dataset_cat[1]
        if show_rbf
            df_rbf = @rsubset(df_group, :kernel_cat == "RadialBasis")
            if !isempty(df_rbf)
                hline = plot_rbf(ax, df_rbf, measure, measure_type, show_std=show_bands)
                interactive && connect!(hline.visible, toggles_dict["RadialBasis"].active)
            end
        end
        ax.title = string(dataset)
    end

    if interactive
        foreach(zip(slines, plots_kernels)) do (sline, (_, _, kernel))
            connect!(sline.visible, toggles_dict[kernel].active)
        end

        foreach(bands) do band
            connect!(band.visible, bands_toggle.active)
        end
    end

    # Get string representation of measure and resampling for the labels
    resampling = string(df.resampling[1])
    measure_name = name(measure_type)

    Label(fig[1, 0], text=measure_name, font=:bold, fontsize=17, tellheight=false, rotation=pi / 2)
    Label(fig[2, 1], text=L"Sigma ($\sigma_w$)", font=:bold, fontsize=17, tellwidth=false)
    Label(fig[0, 1], text="Sigma vs $measure_name by Dataset ($resampling)", font=:bold, fontsize=20, tellwidth=false)

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
    rowgap!(gr, 10)
    colgap!(gr, 10)

    if !interactive
        tellwidth = !vertical
        tellheight = vertical

        (n, m) = size(gr)

        pos = if n * m > length(datasets) # if there is a cell available, use that
            vertical = false
            tellwidth = false
            tellheight = false

            gr[n, m]
        elseif vertical
            fig[end+1, 1]
        else
            fig[1, 2]
        end
        Legend(pos, axes[1], "Kernels"; merge=true, framevisible=options.framevisible,
            orientation=ifelse(vertical, :horizontal, :vertical),
            tellwidth, tellheight)
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

function plot_frenay()
    fig = Figure()

    frenay_small = sort!(CSV.read(datadir("frenay_table2.csv"), DataFrame), :sigma)
    frenay_large = sort!(CSV.read(datadir("frenay_table3.csv"), DataFrame), :sigma)

    rbf_large = frenay_large[frenay_large.kernel.=="RBF", :]
    rbf_small = frenay_small[frenay_small.kernel.=="RBF", :]

    asin = data(vcat(frenay_large, frenay_small) |> dropmissing) *
           mapping(:sigma, :mse => "MSE", lower=:lower, upper=:upper, color=:kernel) *
           (visual(LinesFill) + visual(Scatter))

    facet = mapping(layout=:dataset)

    rbf = data(vcat(rbf_large, rbf_small)) * (
        (mapping(:upper) + mapping(:lower)) *
        visual(HLines, color=options.rbf_color, linestyle=options.rbf_linestyle) +
        mapping(:mse => "MSE") * visual(HLines, color=options.rbf_color)
    )

    plt = (rbf + asin) * facet

    grid = draw!(fig, plt, facet=(; linkyaxes=:none), axis=(; xscale=log10))

    # Work around to put the legend on the bottom right and have a custom
    # Design for RBF
    legend = AlgebraOfGraphics.compute_legend(grid)

    entry1 = legend[1][1]

    entry2 = [
        LineElement(color=options.rbf_color, linestyle=options.rbf_linestyle, points=Point2f[(0, 0.75), (1, 0.75)])
        LineElement(color=options.rbf_color)
        LineElement(color=options.rbf_color, linestyle=options.rbf_linestyle, points=Point2f[(0, 0.25), (1, 0.25)])
    ]

    Legend(fig[3, 3], [entry1, entry2], ["AsinNorm", "RBF"], "Kernel", tellwidth=false, tellheight=false, framevisible=options.framevisible)

    Label(fig[end+1, :], L"Sigma ($\sigma$)", fontsize=17)

    fig
end
