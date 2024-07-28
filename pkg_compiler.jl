using PackageCompiler

create_sysimage(
    ["Flux", "NNlib", "Optim", "CUDA", "cuDNN", "LinearAlgebra", "KernelAbstractions", "Tullio", "SymPy", "Test", "Random", "Statistics", "GLM", "DataFrames", "Optimisers", "PyCall", "Zygote", "ProgressBars", "CSV", "Conda", "Dates", "Printf", "Makie", "GLMakie", "FileIO", "IntervalSets"],
    sysimage_path="precompile.so", precompile_execution_file="precompile.jl"
    )

# julia --sysimage precompile.so file.jl