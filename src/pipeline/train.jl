using Flux, CUDA, KernelAbstractions, Optim

include("../utils.jl")
using .Utils: create_loaders

struct trainer
    train_loader::Flux.Data.DataLoader
    test_loader::Flux.Data.DataLoader
end

function init_trainer(fcn, train_loader, test_loader; opt="lbfgs")
    
end

