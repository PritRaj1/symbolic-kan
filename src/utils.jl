module Utils

export device

using Flux, CUDA

const USE_GPU = CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))

function device(x)
    return USE_GPU ? gpu(x) : x
end

end