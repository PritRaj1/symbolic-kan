module PipelineUtils

export create_loaders, log_csv, L2_loss!

using Flux, Statistics, Random
# using CUDA, KernelAbstractions

include("../architecture/kan_model.jl")
using .KolmogorovArnoldNets: fwd!

function L2_loss!(model, x, y)
    """
    Compute L2 loss between predicted and true values.
    
    Args:
    - model: KAN model.
    - x: input tensor.
    - y: true output tensor.
    
    Returns:
    - loss: L2 loss.
    """
    ŷ = fwd!(model, x)
    return mean(sum((ŷ .- y).^2))
end

function diff3(A)
    s1, s2, s3 = size(A)
    return [A[i,j,k+1] - A[i,j,k] for i=1:s1, j=1:s2, k=1:s3-1]
end

# Log the loss to CSV
function log_csv(epoch, time, train_loss, test_loss, reg, file_name; log_time=true)
    open(file_name, "a") do file
        log_time ? write(file, "$epoch,$time,$train_loss,$test_loss,$reg\n") : write(file, "$epoch,$train_loss,$test_loss,$reg\n")
    end
end

### Data loaders ###
function create_loaders(fcn; N_var=2, x_range=(-1.0,1.0), N_train=1000, N_test=1000, batch_size=32, normalise_input=false, init_seed=nothing)
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
    X_train = randn(Float32, (N_var, N_train)) .* (x_range[2] - x_range[1]) .+ x_range[1]
    X_test = randn(Float32, (N_var, N_test)) .* (x_range[2] - x_range[1]) .+ x_range[1]

    # Normalise data
    if normalise_input
        X_train = (X_train .- mean(X_train, dims=1)) ./ std(X_train, dims=1)
        X_test = (X_test .- mean(X_test, dims=1)) ./ std(X_test, dims=1)
    end

    y_train = zeros(Float32, 1, 0)
    y_test = zeros(Float32, 1, 0)

    for i in 1:N_train
        y_train = hcat(y_train, fcn(X_train[:, i]))
    end
    for i in 1:N_test
        y_test = hcat(y_test, fcn(X_test[:, i]))
    end

    # Create dataloaders
    train_loader = Flux.Data.DataLoader((X_train, y_train); batchsize=batch_size)
    test_loader = Flux.Data.DataLoader((X_test, y_test); batchsize=batch_size)

    return train_loader, test_loader
end

end
