# symbolicKAN

Julia implementation of B-spline KAN for symbolic regression - recreated pretty much as is from [pykan](https://github.com/KindXiaoming/pykan).

## Intro

The purpose of recreating pykan in Julia was to deepen my own understanding of KANs. I'm personally very interested in symbolic regression and scientific machine learning. 

AI is more than just chatbots, productivity boosters, and targeted advertising. We hold the potential means of parameterising or learning any mathematically conceivable system in the universe, (given sufficient hardware, data, and time), but for some reason, everyone is fixated on sentence generation.

If you want to model climate change, water engineering, electromagnetics, molecular dynamics, or mechanical vibrations, you can't just disgrgard centuries worth of accumulated human knowledge and chuck another Transformer at the problem - you're going to have to be a bit more scientific; and the next turning point in AI isn't going to be infinite context windows or automated customer service. 

Those same sequence modelling techniques used in NLP could instead be applied towards [predicting non-linear material deformation](https://github.com/exa-laboratories/Julia-Wav-KAN). On a similar note, the main ideas from image generation could instead be used to [model 2D fluid flow through porous media](https://github.com/exa-laboratories/wavKAN-conv2D).

Overall, I really think that KANs represent something bigger - a tool towards unravelling the formulation underpinning a dataspace, and offering up new pathways towards expanding human knowledge, refining architectures and priors, and working with limited and noisy scientific data.   

KindXiaoming and everyone else in the KAN community have set the stage for something legendary, and I will always be grateful to them for putting it out there for the world to see.

Thanks,
Prit

## TODO

1. CUDA
2. mutable structs need to disappear -> replace with @set from Accessors.jl
3. Optim for L-BFGS