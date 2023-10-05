import ..Utils
import ..Experiments
import ..DataSets

using DataFramesMeta
using CategoricalArrays

import MLJBase: skipnan

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
    groups = @chain df begin
        @orderby(:is_delve, :dataset_cat)
        groupby(:dataset_cat, sort=false)
    end

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
    show_grid=true,
    show_text=false,
    show_labels=false,
    num_breaks=5,
    alpha=nothing,
    categorical=false,
    kwargs...)

    fig = Figure(; kwargs...)
    ax_opts = ()
    mm = build_matrices_by_dataset(df; kernel_l, kernel_r, sigma, measure)

    if !isnothing(alpha)
        # Remove non statistically significant values
        if measure == :per_fold
            @warn "Alpha is ignored when measure is :per_fold"
            alpha = nothing
        else
            pmatrices = build_matrices_by_dataset(df; kernel_l, kernel_r, sigma, measure=:per_fold)

            foreach(zip(mm, pmatrices)) do ((_, _, mat), (_, _, pmat))
                for i in eachindex(mat)
                    if pmat[i] > alpha
                        mat[i] = NaN
                    end
                end
            end
        end
    end

    extremas = map(mm) do (_, _, mat)
        nonnan_ = skipnan(mat)
        if isempty(nonnan_)
            @warn "Empty matrix"
            (0, 0)
        else
            extrema(nonnan_)
        end
    end
    vmax = round(max(abs.(minimum(first.(extremas))), maximum(last.(extremas))), sigdigits=1)

    if measure == :per_fold # we are ploting p-values
        colorrange = (0, 1)
        colormap = :devon
        cb_label = L"$p$-value"
        categorical = true
    else
        colorrange = (-vmax, vmax)
        if isnan(vmax)
            @warn "Empty colorrange"
            colorrange = (-1, 1)
        end
        colormap = Reverse(:vik)
        cb_label = L"$\Delta$nRMSE (%$(kernel_l) - %$(kernel_r))"
    end

    gr, axes, (m, _) = build_grid(fig[1, 1], length(mm), dims; ax_opts...)

    foreach(zip(axes, mm)) do (ax, (key, labels, matrix))
        ax.title = key[1] |> string
        plot_heatmap(key, matrix, labels; kernel_l, kernel_r, fig, ax, colorrange, colormap, categorical, show_grid, show_text, show_labels, num_breaks)
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

    colgap!(gr, 10)
    rowgap!(gr, 10)

    variable_l = ifelse(kernel_l == :RadialBasis, "gamma", "sigma")
    variable_r = ifelse(kernel_r == :RadialBasis, "gamma", "sigma")
    Label(fig[1:end, 0], L"%$(kernel_l) ($\%$variable_l$)", rotation=pi / 2, tellheight=false, fontsize=21)
    Label(fig[end+1, 1:end], L"%$(kernel_r) ($\%$variable_r$)", tellwidth=false, fontsize=21)

    if measure == :per_fold
        _, break_labels = p_breaks(num_breaks)
        n_categories = length(break_labels)
        colormap = cgrad(colormap, n_categories, categorical=true)
        # label_pos = ([1 .. n_categories] .- 0.5) ./ n_categories
        label_pos = (range(1, n_categories, step=1) .- 0.5) ./ n_categories
        Colorbar(fig[end+1, 1:end], limits=colorrange, colormap=colormap, label=cb_label, vertical=false, ticks=(label_pos, break_labels), flipaxis=false, labelsize=21)
    else
        Colorbar(fig[end+1, 1:end], limits=colorrange, colormap=colormap, label=cb_label, vertical=false, flipaxis=false, labelsize=21)
    end


    fig
end

function plot_heatmap(mat, args...; kernel_l=:RadialBasis, kernel_r=:AsinNorm, kwargs...)
    title = mat[1][1] |> string
    matrix = mat[3]
    labels = mat[2]
    plot_heatmap(title, matrix, labels, args...; kernel_l, kernel_r, kwargs...)
end

function ten_exp_to_string(n)
    if n < 0
        return "0." * repeat("0", abs(n) - 1) * "1"
    end
    return "1" * repeat("0", n)
end

function p_breaks(n)
    # breaks = [0, 0.001, 0.01, 0.1, 1]
    exps = range(-n, 0)
    breaks = 10.0 .^ exps
    breaks = [0, breaks...]

    breaks_str = ten_exp_to_string.(exps)
    labels = [L"<%$(x)" for x in breaks_str[2:end-1]]
    labels = append!(labels, [L"\geq%$(breaks_str[end-1])"])

    breaks, labels
end

# WARN: this is a mess of x and y. Pretty sure it is correct. But should not
# mess with it.
function plot_heatmap(title, matrix, labels, args...; kernel_l, kernel_r, fig=Figure(),
    ax=Axis(fig[1, 1]; title),
    colorrange=nothing,
    colormap=Reverse(:vik),
    categorical=false,
    show_grid=true,
    show_text=!categorical,
    show_labels=true,
    num_breaks=5,
    kwargs...)
    ylabel = ifelse(kernel_l == :RadialBasis, "gamma", "sigma")
    ylabel = "$kernel_l ($ylabel)"
    xlabel = ifelse(kernel_r == :RadialBasis, "gamma", "sigma")
    xlabel = "$kernel_r ($xlabel)"

    ax.ylabel = ylabel
    ax.xlabel = xlabel

    ax.xlabelvisible = show_labels
    ax.ylabelvisible = show_labels

    doColorbar = isnothing(colorrange)

    if doColorbar
        valmax = maximum(abs.(skipnan(matrix)))
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

    if categorical
        breaks, break_labels = p_breaks(num_breaks)
        mat_cat_s = cut(replace(matrix, NaN => 1), breaks, extend=true)
        n_categories = length(break_labels)
        mat_cat = (levelcode.(mat_cat_s) .- 0.5) ./ n_categories
        colorrange = (0, 1)
        colormap = cgrad(colormap, n_categories, categorical=true)
    end

    exp_x = round.(Int, log10.(xs))
    exp_y = round.(Int, log10.(ys))

    x_labels = map(x -> L"10^{%$(x)}", exp_x)
    y_labels = map(x -> L"10^{%$(x)}", exp_y)


    # x_ticks = 1:length(xs)
    # y_ticks = 1:length(ys)
    x_ticks = exp_x
    y_ticks = exp_y

    ax.yticks = (x_ticks[1:3:end], x_labels[1:3:end])
    ax.xticks = (y_ticks[1:3:end], y_labels[1:3:end])

    n = length(xs)
    m = length(ys)

    # ax2 = Axis(fig[1, 2], title="sigma", xlabel="gamma", ylabel="sigma")
    # hm = heatmap!(ax, 1:7, 1:11, matrix, colormap=:RdBu, colorrange=colorrange)
    if categorical
        hm = heatmap!(ax, exp_y, exp_x, mat_cat; colormap, colorrange)
    else
        hm = heatmap!(ax, exp_y, exp_x, matrix; colormap, colorrange)
    end

    if show_grid
        # Move heatmap to the back so that the grid is visible
        translate!(hm, 0, 0, -100)

        yticks = exp_y .+ 0.5
        xticks = exp_x .+ 0.5

        append!(yticks, minimum(exp_y) - 0.5)
        append!(xticks, minimum(exp_x) - 0.5)

        ax.xminorticks = yticks
        ax.yminorticks = xticks

        ax.xminorgridvisible = true
        ax.xgridcolor = :transparent
        ax.xminorgridcolor = RGBAf(0.5, 0.5, 0.5, 0.5)
        ax.yminorgridvisible = true
        ax.ygridcolor = :transparent
        ax.yminorgridcolor = RGBAf(0.5, 0.5, 0.5, 0.5)
    end

    values = map(string.(round.(matrix, digits=2))) do x
        if x == "NaN"
            ""
        else
            x
        end
    end

    show_text && text!(ax, values[:],
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
