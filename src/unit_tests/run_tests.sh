#!/bin/bash

TEST_DIR="src/unit_tests"

test_files=$(find "$TEST_DIR" -name "*.jl")

# Run each test file
for test_file in $test_files; 
do
    echo "Running $test_file"
    julia "$test_file"
    if [ $? -ne 0 ]; then
        echo "Test failed: $test_file"
        exit 1
    fi
done