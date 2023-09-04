#!/usr/bin/env julia

using TFM.DataSets
using CSV

datasets = DataSets.all
output_folder = get(ENV, "DEVENV_ROOT", ".") |> x -> joinpath(x, "data", "datasets_processed")

eprintln(x...) = println(stderr, x...)

if output_folder == "."
    eprintln("No output folder specified, using current folder")
else
    eprintln("Saving datasets to: ", output_folder)
end

mkpath(output_folder)

for dataset in datasets
    eprintln("Processing dataset: ", dataset)
    X, y = DataSets.unpack(dataset)

    X_file = joinpath(output_folder, string(dataset) * ".X")
    y_file = joinpath(output_folder, string(dataset) * ".y")

    CSV.write(X_file, X)

    # y is a vector, so we need just dump it as newline separated values
    open(y_file, "w") do f
        for yi in y
            println(f, yi)
        end
    end
end
