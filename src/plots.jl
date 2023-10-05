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

const options = (;
    band_opacity=0.3,
    rbf_color=:red,
    rbf_linestyle=:dot,
    rbf_linewidth=2,
    framevisible=true # Show legend frame
)

include("./plots/helpers.jl")
export @saveplot

using AlgebraOfGraphics
using DataFrames, DataFramesMeta, MLJ, DrWatson

include("./plots/kernels.jl")
include("./plots/model_performance.jl")
include("./plots/execution_times.jl")
include("./plots/heatmap.jl")

end
