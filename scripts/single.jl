using DrWatson
@quickactivate "tfm"

# Here you may include files from the source directory
include(srcdir("helpers.jl"))
include(srcdir("datasets.jl"))

println(
    """
    Currently active project is: $(projectname())
    Path of active project: $(projectdir())
    """
)

df, target = Main.DataSets.cpu()

using MLJ
using LIBSVM.Kernel

yhat, ytest, (mach, Xtrain, ytrain, Xtest, ytest) = run_single(Kernel.Asin, df, target)

println(mean((yhat .- ytest) .^ 2))
