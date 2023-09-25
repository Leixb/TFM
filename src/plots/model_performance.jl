
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

function plot_rbf(ax::Axis, df_rbf::AbstractDataFrame, measure::Symbol, measurement_type::Type{<:MLJBase.Measure})
    # Make sure all rows are "RadialBasis"
    @assert allequal(df_rbf.kernel)
    @assert df_rbf.kernel[1] |> string == "RadialBasis"

    best = summarizer(measurement_type)(getproperty(df_rbf, measure))
    hlines!(ax, [best], label="RBF (best)", color=:red, linestyle=:dot, linewidth=2)
end


function plot_sigma_band(ax, df_sub_kern, kernel; sigma::Symbol, measure::Symbol, std::Symbol=:std)
    val_sigma = getproperty(df_sub_kern, sigma)
    val_lower = getproperty(df_sub_kern, measure) .- getproperty(df_sub_kern, std)
    val_upper = getproperty(df_sub_kern, measure) .+ getproperty(df_sub_kern, std)

    band!(
        ax,
        val_sigma, val_lower, val_upper,
        label=kernel |> string, visible=true,
        color=(kernel_color(kernel), 0.3)
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

    measure_type = typeof(df.measure[1])

    # Make sure we are not mixing measures
    @assert allequal(df.measure)
    @assert measure_type <: MLJBase.Measure

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
    with_theme(
        ScatterLines=(cycle=Cycle([:color, :marker, :linestyle], covary=true),)
    ) do
        # We plot bands first so they are on bottom
        show_bands && foreach(plot_band, plots_and_kernels)

        foreach(plot_kernel, plots_and_kernels)

        show_rbf && foreach(subplots) do (ax, df_sub)
            df_rbf = @rsubset(df_sub, :kernel_cat == "RadialBasis")
            if !isempty(df_rbf)
                plot_rbf(ax, df_rbf, measure, measure_type)
            end
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
    Legend(fig[1:2, 4], ax.fm, "Kernel", framevisible=false, merge=true)
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

function build_grid(pos::GridPosition, len::Int, dims::Union{Nothing,Tuple{Int,Int}}=nothing)
    gr = GridLayout(pos)

    if isnothing(dims)
        n = Int(ceil(sqrt(len)))
        m = Int(ceil(len / n))
    else
        (m, n) = dims
    end

    axes = [Axis(gr[i, j]) for j in 1:m, i in 1:n if (i - 1) * m + j <= len]

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
    kwargs...
)

    # TODO: pass as parameter?
    fig = Figure(args...; kwargs...)

    # Remove kernels that we won't show, so that we don't run into
    # issues with empty plots.
    df_filtered = filter(df) do row
        row.kernel_cat in show_kernels
    end

    measure_type = typeof(df_filtered.measure[1])

    # Make sure we are not mixing measures
    @assert allequal(df_filtered.measure)
    @assert measure_type <: MLJBase.Measure

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

    gr, axes, (m, _) = build_grid(fig[1, 1], length(datasets), dims)

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

                bands = plot_sigma_band(ax, df_kernel, kernel; sigma, measure, std)

                if interactive
                    connect!(bands.visible, bands_toggle.active)
                end
            end
        end

        for df_kernel in df_subgroup
            kernel = df_kernel.kernel_cat[1]

            if kernel == "RadialBasis"
                if show_rbf || interactive
                    hline = plot_rbf(ax, df_kernel, measure, measure_type)
                    interactive && connect!(hline.visible, toggles_dict[kernel].active)
                end
                continue
            elseif !(kernel in show_kernels || interactive)
                continue
            end

            slines = plot_sigma_kernel(ax, df_kernel, kernel; sigma, measure)

            if interactive
                connect!(slines.visible, toggles_dict[kernel].active)
            end
        end
        ax.title = string(dataset)
        ax.xscale = log10
    end

    # Get string representation of measure and resampling for the labels
    resampling = string(df.resampling[1])
    measure_name = name(measure_type)

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
