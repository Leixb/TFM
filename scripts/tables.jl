using TFM
using PrettyTables

all_ds = TFM.DataSets.all

df = let
    df = DataFrame(
        name=String[],
        doi=String[],
        #url=String[],
        rows=Int[],
        cols=Int[],
        kind=String[],
    )


    for ds in all_ds
        if ds == DataSets.MNIST()
            continue
        end

        doi = DataSets.doi(ds)
        if isnothing(doi)
            doi = ""
        end

        X = DataSets.data(ds)
        push!(df, (
            string(ds),
            doi,
            #DataSets.url(ds),
            nrow(X),
            ncol(X),
            DataSets.is_regression(ds) ? "Regression" : "Classification",
        ))
    end

    df
end

aligns = map(x -> x === String ? :c : :r, eltype.(eachcol(df)))
aligns[1] = :l

open("./document/tables/datasets.tex", "w") do io
    pretty_table(io, df, header=names(df), backend=Val(:latex), tf=tf_latex_booktabs, alignment=aligns)
end
