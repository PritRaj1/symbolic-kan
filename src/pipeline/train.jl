using Flux, CUDA, KernelAbstractions, Optim

include("utils.jl")
using .PipelineUtils

struct trainer
    train_loader::Flux.Data.DataLoader
    test_loader::Flux.Data.DataLoader
    
end

function init_trainer(model, train_loader, test_loader, optimiser, scheduler, loss_fn; max_epochs=100, verbose=true)
    """
    Initialise trainer for training symbolic model.

    Args:
    - model: symbolic model to train.
    - train_loader: training dataloader.
    - test_loader: test dataloader.
    - optimiser: optimiser to use.
    - scheduler: learning rate scheduler.
    - loss_fn: loss function.
    - max_epochs: maximum number of epochs.
    - verbose: whether to print training progress.

    Returns:
    - t: trainer object.
    """
    
end
    


