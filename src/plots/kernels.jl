import ..Utils: kernel_asin_normalized

using LaTeXStrings

"Plot the kernel function around the origin with different values of σ"
function plot_kernel(kernel=kernel_asin_normalized, args...;
    fig_opts::NamedTuple=(backgroundcolor = :transparent),
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
        fig[1, 2] = Legend(fig, ax, L"\sigma_w", framevisible=options.framevisible)
    end

    ylims!(ax, 0.5, 1)
    xlims!(ax, extrema(x))

    fig
end

function plot_kernel_3d_grid(kernel, args...; offset=pi / 5, dims=(2, 2), kwargs...)
    fig = Figure(backgroundcolor=:transparent)
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
    fig = Figure(backgroundcolor=:transparent)
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

