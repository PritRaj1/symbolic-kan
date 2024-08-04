module PipelineUtils

export create_data, log_csv

using Statistics, Random
using CUDA, KernelAbstractions

# Log the loss to CSV
function log_csv(epoch, time, train_loss, test_loss, reg, file_name; log_time=true)
    open(file_name, "a") do file
        log_time ? write(file, "$epoch,$time,$train_loss,$test_loss,$reg\n") : write(file, "$epoch,$train_loss,$test_loss,$reg\n")
    end
end

### Data ###
function create_data(fcn; N_var=2, x_range=(-1.0,1.0), N_train=1000, N_test=1000, normalise_input=false, init_seed=nothing)
    """
    Create train and test dataloaders

    Args:
    - fcn: symbolic function to generate data for.
    - N_var: number of input variables.
    - range: range of input variables.
    - N_train: number of training samples.
    - N_test: number of test samples.
    - normalise_input: whether to normalise input.
    - normalise_output: whether to normalise output.
    - init_seed: random seed.

    Returns:
    - train_loader: training dataloader.
    - test_loader: test dataloader.
    """

    Random.seed!(init_seed)

    # Generate data
    X_train = randn(Float32, (N_train, N_var)) .* (x_range[2] - x_range[1]) .+ x_range[1]
    X_test = randn(Float32, (N_train, N_var)) .* (x_range[2] - x_range[1]) .+ x_range[1]

    # Normalise data
    if normalise_input
        X_train = (X_train .- mean(X_train, dims=1)) ./ std(X_train, dims=1)
        X_test = (X_test .- mean(X_test, dims=1)) ./ std(X_test, dims=1)
    end

    y_train = zeros(Float32, 0, 1)
    y_test = zeros(Float32, 0, 1)

    for i in 1:N_train
        y_train = vcat(y_train, fcn(X_train[i, :]))
    end
    for i in 1:N_test
        y_test = vcat(y_test, fcn(X_test[i, :]))
    end

    return (X_train, y_train), (X_test, y_test)
end

end
