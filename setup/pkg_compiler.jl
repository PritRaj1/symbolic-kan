using PackageCompiler

create_sysimage(
    [
        "Lux", "Flux", "NNlib", "OptimizationOptimJL", "CUDA", "cuDNN", "LinearAlgebra", "KernelAbstractions",
        "Tullio", "SymPy", "Test", "Random", "Statistics", "GLM", "DataFrames", "Optimisers", "PyCall",
        "Zygote", "ProgressBars", "CSV", "Dates", "Printf", "Makie", "GLMakie", "FileIO","IntervalSets", 
        "LineSearches", "FluxOptTools", "Plots", "DifferentialEquations", "Accessors", "Optimisers", 
        "ComponentArrays", "ConfParser", "LuxCUDA", "SpecialFunctions", "LaTeXStrings", "ProfileView",
        "MAT", "Optimization", "HypothesisTests"
    ],
    sysimage_path="precompile.so",
    precompile_execution_file="setup/precompile.jl"
)