# Helper functions for plotting

using DrWatson: projectdir
using Makie, MathTeXEngine
using MLJBase, MLJ
using DataFrames, DataFramesMeta

import ..DataSets: is_regression
import ..Measures, ..DataSets

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

"Get a DataFrame with the best result for the given grouping and measurement"
function summarize_best(df::DataFrame, grouping::Array{Symbol}=[:dataset_cat, :kernel_cat]; by::Symbol=:measurement, maximum=false)
    # To get the best result, we first sort by the measurement and then
    # take the first row of each group
    # If maximum, we reverse the order of the sort
    @chain df begin
        sort(by, rev=maximum)
        groupby(grouping)
        combine(first, nrow)
    end
end

"Filter the DataFrame to get only the regression datasets"
regression(df::DataFrame) = @rsubset(df, is_regression(:dataset))

"Filter the DataFrame to get only the classification datasets"
classification(df::DataFrame) = @rsubset(df, !is_regression(:dataset))

function kernel_idx(kernel::String)
    kernels = ["AsinNorm", "Asin", "Acos0", "Acos1", "Acos2", "RadialBasis"]
    idx = findfirst(kernels .== kernel)
    if isnothing(idx)
        @warn "Unknown kernel: $kernel"
        return 0
    end
    return idx
end

function kernel_color(kernel::String; colors=Makie.wong_colors())
    # if kernel == "RadialBasis" return :red end
    return colors[kernel_idx(kernel)]
end

kernel_idx(kernel) = kernel_idx(string(kernel))
kernel_color(kernel; kwargs...) = kernel_color(string(kernel); kwargs...)

getindex(::typeof(kernel_idx), kernel::String) = kernel_idx(kernel)
getindex(::typeof(kernel_color), kernel::String) = kernel_color(kernel)

function get_measure_type(df::AbstractDataFrame)::Type{<:MLJBase.Measure}
    measure_type = typeof(df.measure[1])

    # Make sure we are not mixing measures
    @assert allequal(df.measure)
    @assert measure_type <: MLJBase.Measure

    measure_type
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

# We want to minimize MSE and maximize Accuracy
is_maximize(::Type{<:MLJBase.Measure}) = false
is_maximize(::Type{MLJ.Accuracy}) = true

# Convenience functions to Name different types consistently
name(m::Type{<:MLJBase.Measure}) = m |> string |> titlecase
name(::Type{Measures.MSE}) = "MSE"
name(::Type{Measures.nRMSE}) = "nRMSE"
name(ds::Type{<:DataSets.DataSet}) = split(string(ds), '.') |> last

"""
Convenience function to wrap a function so that it can be used with
map and foreach tuples.

It takes a function of the form `fn(x, y, z...)` and returns a function
of the form `fn((x, y, z...))`. Notice that the function returned takes
a single argument, a tuple, which is then unpacked and passed to the original
function.
"""
wrap_tuple = fn -> (x -> fn(x...))

"""Useful for faceting, allows to create a box with a label inside"""
function BoxLabel(fig::GridPosition, args...; padding=(5, 5, 5, 5), boxopts=(;), kwargs...)
    Box(fig; boxopts...)
    Label(fig, args...; padding, kwargs...)
end

"""
Sets the CreationDate of the pdf metadata to zero, so that the file
does not change when the plot is saved again.

WARNING: It is important that timestamp is a string of length 20, otherwise
the pdf metadata will be corrupted.

WARNING: This modifies the file in-place, use with caution.
"""
function __strip_metadata(filename::String, timestamp::String="00000000000000+00'00")
    @assert length(timestamp) == 20 "Timestamp must be a string of length exactly 20"
    run(`sed -i "s/CreationDate.*$/CreationDate (D:$timestamp)/" $filename`, wait=true)
end

plotsdocdir(args...) = projectdir("document", "figures", "plots", args...)

"""
# Helper macro to save plots quickly.

It saves the plot in the document/figures/plots folder with the name of
the variable given to the plot

## Example

```julia
using Plots

x = 1:10
y = rand(10)

@saveplot myscatter = scatter(x, y)
```

This will save the plot in document/figures/plots/myscatter.pdf
"""
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

num_features = Dict(
    "LiverDisorders" => 5,
    "AutoMpg" => 7,
    "BikeSharingHour" => 16,
    "Abalone" => 8,
    "Stock" => 9,
    "CPU" => 6,
    "Ionosphere" => 34,
    "Pumadyn8nm" => 8,
    "GlassIdentification" => 9,
    "CompActs" => 21,
    "CancerDiagnostic" => 30,
    "StatlogGermanCreditData" => 20,
    "Bank32nm" => 32,
    "Pumadyn8fh" => 8,
    "Pumadyn32nm" => 32,
    "Pumadyn32nh" => 32,
    "Bank8nh" => 8,
    "Adult" => 14,
    "Pumadyn32fm" => 32,
    "Pumadyn8fm" => 8,
    "Hepatitis" => 19,
    "Wine" => 13,
    "BikeSharingDay" => 15,
    "Bank8fh" => 8,
    "CommunitiesAndCrime" => 127,
    "Pumadyn32fh" => 32,
    "Elevators" => 6,
    "Servo" => 4,
    "Bank32nh" => 32,
    "Ailerons" => 7,
    "Bank32fm" => 32,
    "Pumadyn8nh" => 8,
    "Iris" => 4,
    "Bank32fh" => 32,
    "Triazines" => 58,
    "Bank8fm" => 8,
    "EatingHabits" => 15,
    "EnergyEfficiencyCooling" => 8,
    "EnergyEfficiencyHeating" => 8,
    "Cancer" => 32,
    "MNIST" => 784,
    "Bank8nm" => 8,
)
