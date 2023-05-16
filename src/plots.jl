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

using CairoMakie, GLMakie, LaTeXStrings
using DrWatson

import ..Utils

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

function plot_kernel_3d(kernel, args...; offset = pi/5, dims=(2, 2), kwargs...)
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

end
