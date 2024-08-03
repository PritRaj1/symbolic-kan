#!/bin/bash

# Run requirements.jl
echo "Running setup..."
julia "./requirements.jl"
if [ $? -ne 0 ]; then
    echo "Failed to run requirements.jl"
    exit 1
fi

# Run pkg_compiler.jl
echo "Precompiling packages..."
julia "./pkg_compiler.jl"
if [ $? -ne 0 ]; then
    echo "Failed to run pkg_compiler.jl"
    exit 1
fi

echo "Both Julia scripts ran successfully."
