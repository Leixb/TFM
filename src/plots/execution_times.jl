
using CSV

import ..Benchmark

function data_benchmark_meta()::DataFrame
    df = Benchmark.Flattened(datadir("benchmark_multi.json"))
    meta = CSV.read(datadir("model_sizes.csv"), DataFrame)

    leftjoin!(df, meta; on=:model => :dataset)
    df.kind = ifelse.(startswith.(df.model, "a"), "Adult", "Webdata")

    df
end

function plot_benchmark_time_instances(df::DataFrame=data_benchmark_meta())
    df = @chain df begin
        groupby([:kernel, :instances, :kind])
        combine(:time => mean => :time, :time => std => :std)
    end

    sr = x -> sorter(["Asin", "AsinNorm", "Acos0", "Acos1", "Acos2", "RadialBasis"])(string(x))

    cols = mapping(
        :instances,
        #:binary => "Binary",
        :time => "Execution time (s)",
        # lower=(:time, :std) => -,
        # upper=(:time, :std) => +,
        color=:kernel => sr => "Kernel",
        #dodge=:model
    )
    grp = mapping(layout=:kind)
    # geom = visual(LinesFill) + visual(Scatter)# + visual(BoxPlot, width=0.1)
    geom = visual(ScatterLines)

    plt = data(df) * cols * geom * grp
    fg = draw(plt,
        facet=(; linkyaxes=true, linkxaxes=false),
        axis=(;
            # xticklabelrotation=pi / 4,
            #limits=((nothing, nothing), (0, nothing)),
            yscale=log10,
            xscale=log10
        ),)

    fg
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
