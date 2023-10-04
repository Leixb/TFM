import ..Utils
import ..Experiments

using DataFramesMeta

# FIX: move this to a more sensible place and use it in the other files
function data_nrmse_s()
    @chain Experiments.experiment_data("svms3", false) begin
        summarize_best([:kernel_cat, :dataset_cat, :sigma])
        regression()
    end
end

function data_heatmap(df_heat::AbstractDataFrame=data_nrmse_s(); kernel_l=:RadialBasis, kernel_r=:AsinNorm, measure=:measurement)
    nrmse_wide = unstack(df_heat, [:dataset_cat, :sigma, :sigma_scaled], :kernel_cat, measure)

    function get_df(kernel_name::Symbol)
        @chain nrmse_wide begin
            dropmissing(kernel_name)
            @transform(:gamma = Utils.sigma2gamma(:sigma))
            @select(:dataset_cat, :sigma, $(kernel_name), :gamma, :sigma_scaled)
            @rename(:value = $(kernel_name))
            @orderby(:sigma)
        end
    end

    get_df(kernel_l), get_df(kernel_r)
end

function build_matrix(df1::AbstractDataFrame, df2::AbstractDataFrame; col=:value, kernel_l=:RadialBasis, kernel_r=:AsinNorm, sigma=:sigma, args...)
    # combine the two columns and do a matrix with the differences
    mat = zeros(size(df1, 1), size(df2, 1))

    xs = ifelse(kernel_l == :RadialBasis, df1.gamma, getproperty(df1, sigma))
    ys = ifelse(kernel_r == :RadialBasis, df1.gamma, getproperty(df2, sigma))

    for (i, row1) in enumerate(eachrow(df1))
        for (j, row2) in enumerate(eachrow(df2))
            if row1[col] isa AbstractArray
                mat[i, j] = Experiments.paired_ttest_5x2cv(row1[col], row2[col])[2] # (t-stat, p-value)[1] -> t-stat
            else
                mat[i, j] = row1[col] - row2[col]
            end
        end
    end

    xs, ys, mat
end

function build_matrices_by_dataset(df=data_nrmse_s(); kernel_l=:RadialBasis, kernel_r=:AsinNorm, measure=:measurement, kwargs...)
    groups = groupby(df, :dataset_cat)

    matrices = []
    labels = []

    foreach(groups) do group
        df1, df2 = data_heatmap(group; kernel_l, kernel_r, measure)
        xs, ys, m = build_matrix(df1, df2; kernel_l, kernel_r, kwargs...)
        push!(matrices, m)
        push!(labels, (xs, ys))
    end

    zip(keys(groups), labels, matrices)
end

function plot_all_heatmaps(df=data_nrmse_s(); kernel_l=:RadialBasis, kernel_r=:AsinNorm, sigma=:sigma, measure=:measurement,
    dims::Union{Nothing,Tuple{Int,Int}}=nothing,
    linkxaxes=true,
    linkyaxes=true,
    kwargs...)

    fig = Figure(; kwargs...)
    ax_opts = ()
    mm = build_matrices_by_dataset(df; kernel_l, kernel_r, sigma, measure)

    extremas = map(mm) do (_, _, mat)
        extrema(mat)
    end
    vmax = max(abs.(minimum(first.(extremas))), maximum(last.(extremas)))

    colorrange = (-vmax, vmax)

    gr, axes, (m, _) = build_grid(fig[1, 1], length(mm), dims; ax_opts...)

    foreach(zip(axes, mm)) do (ax, (key, labels, matrix))
        ax.title = key[1] |> string
        plot_heatmap(key, matrix, labels; kernel_l, kernel_r, fig, ax, colorrange)
    end

    if linkxaxes
        linkxaxes!(axes...)
        for ax in axes[1:end-m]
            hidexdecorations!(ax; grid=false, minorgrid=false)
        end
    end

    if linkyaxes
        linkyaxes!(axes...)
        for i in (setdiff(Set(1:length(axes)), Set(1:m:length(axes))))
            hideydecorations!(axes[i]; grid=false, minorgrid=false)
        end
    end

    Colorbar(fig[:, end+1], limits=colorrange, colormap=:RdBu, label=L"\Delta nRMSE (%$(kernel_l) - %$(kernel_r))")

    fig
end

function plot_heatmap(mat, args...; kernel_l=:RadialBasis, kernel_r=:AsinNorm, kwargs...)
    title = mat[1][1] |> string
    matrix = mat[3]
    labels = mat[2]
    plot_heatmap(title, matrix, labels, args...; kernel_l, kernel_r, kwargs...)
end

# WARN: this is a mess of x and y. Pretty sure it is correct. But should not
# mess with it.
function plot_heatmap(title, matrix, labels, args...; kernel_l, kernel_r, fig=Figure(),
    ax=Axis(fig[1, 1]; title),
    colorrange=nothing,
    kwargs...)
    ylabel = ifelse(kernel_l == :RadialBasis, "gamma", "sigma")
    ylabel = "$kernel_l ($ylabel)"
    xlabel = ifelse(kernel_r == :RadialBasis, "gamma", "sigma")
    xlabel = "$kernel_r ($xlabel)"

    ax.ylabel = ylabel
    ax.xlabel = xlabel

    doColorbar = isnothing(colorrange)

    if doColorbar
        valmax = maximum(abs.(matrix))
        colorrange = (-valmax, valmax)
    else
        valmax = maximum(abs.(colorrange))
    end

    xs, ys = labels

    # Fix matrix orientation on the plot for RBF (since gamma is reversed)
    matrix = transpose(matrix)
    if kernel_l == :RadialBasis
        matrix = reverse(matrix, dims=2)
        xs = reverse(xs)
    end

    exp_x = round.(Int, log10.(xs))
    exp_y = round.(Int, log10.(ys))

    x_labels = map(x -> L"10^{%$(x)}", exp_x)
    y_labels = map(x -> L"10^{%$(x)}", exp_y)


    # x_ticks = 1:length(xs)
    # y_ticks = 1:length(ys)
    x_ticks = exp_x
    y_ticks = exp_y

    ax.yticks = (x_ticks[1:2:end], x_labels[1:2:end])
    ax.xticks = (y_ticks[1:2:end], y_labels[1:2:end])

    n = length(xs)
    m = length(ys)

    # ax2 = Axis(fig[1, 2], title="sigma", xlabel="gamma", ylabel="sigma")
    # hm = heatmap!(ax, 1:7, 1:11, matrix, colormap=:RdBu, colorrange=colorrange)
    hm = heatmap!(ax, exp_y, exp_x, matrix, colormap=:RdBu, colorrange=colorrange)

    values = string.(round.(matrix, digits=2))

    text!(ax, values[:],
        # position=[Point2f(y, x) for x in 1:n for y in 1:m],
        position=[Point2f(y, x) for x in exp_x for y in exp_y],
        color=ifelse.(matrix .> valmax / 1.5, :white, :black),
        align=(:center, :center),
        fontsize=11
    )

    doColorbar && Colorbar(fig[:, end+1], hm, label="nRMSE")
    # add the labels
    fig
end
