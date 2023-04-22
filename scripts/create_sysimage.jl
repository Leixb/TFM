using PackageCompiler

PackageCompiler.create_sysimage([
        "MLJ",
        "LIBSVM",
        "TFM",
        "Plots",
        "DataFrames",
        "MLJLIBSVMInterface",
    ];
    sysimage_path="sysimage.so",
    precompile_execution_file="precompile_workflow.jl"
)
