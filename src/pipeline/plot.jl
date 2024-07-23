module Plotting

export plot_kan!

using Flux
using Plots
using PyPlot

include("../architecture/kan_model.jl")
using .KolmogorovArnoldNets: prune!

function get_range(model::KAN_, l, i, j; verbose=true)
    """
    Get the range of the activation of neuron (l, i, j) for thresholding.

    Args:
        l: Layer index.
        i: Neuron input index.
        j: Neuron output index.

    Returns:
        x_min: Minimum value of the activation.
        x_max: Maximum value of the activation.
        y_min: Minimum value of the postactivation.
        y_max: Maximum value of the postactivation.
    """
    x, y = model.pre_acts[l][:, i, j], model.post_acts[l][:, i, j]
    x_min, x_max = minimum(x), maximum(x)
    y_min, y_max = minimum(y), maximum(y)
    
    if verbose
        println("x_range: ", x_min, " - ", x_max)
        println("y_range: ", y_min, " - ", y_max)
    end

    return x_min, x_max, y_min, y_max
end

function plot_kan!(model::KAN_; folder="figures/", γ=3, prune_and_mask=false, mode="supervised", σ=0.5, tick=false, sample=false, in_vars=nothing, out_vars=nothing, title=nothing)
    """
    Plot KAN.

    Args:
    - model: KAN model.
    - folder: folder to save plots.
    - γ: controls transparaency of each activation.
    - prune_and_mask: whether to prune and plot mask. true -> prune and plot mask; false -> plot all activations.
    - mode: "supervised" or "unsupervised". "supervised" -> l1 is measured by absolution value; "unsupervised" -> l1 is measured by standard deviation.
    - σ: scale of diagram.
    - tick: whether to show ticks.
    - sample: whether to sample activations.
    - in_vars: input variables.
    - out_vars: output variables.
    - title: title of plot.
    """

    # Create folder if it does not exist
    if !isdir(folder * "splines/")
        mkdir(folder * "splines/")
    end

    depth = length(model.widths) - 1

    for l in 1:depth
        w_large = 2.0
        for i in eachindex(model.widths[l])
            for j in eachindex(model.widths[l+1])
                rank = sortperm(model.acts[l][:, i])

                symbol_mask = model.symbolic_fcns[l].mask[j, i]
                numerical_mask = model.act_fcns[l].mask[i, j]

                # Determine color and alpha_mask based on conditions
                if symbol_mask > 0.0 && numerical_mask > 0.0
                    color = "purple"
                    alpha_mask = 1
                elseif symbol_mask > 0.0 && numerical_mask == 0.0
                    color = "red"
                    alpha_mask = 1
                elseif symbol_mask == 0.0 && numerical_mask > 0.0
                    color = "black"
                    alpha_mask = 1
                else
                    color = "white"
                    alpha_mask = 0
                end

                # Create the plot
                fig = plot(size=(w_large*100, w_large*100))

                if tick
                    x_min, x_max, y_min, y_max = get_range(model, l, i, j; verbose=false)
                    xticks!([x_min, x_max], [@sprintf("%2.f", x_min), @sprintf("%2.f", x_max)])
                    yticks!([y_min, y_max], [@sprintf("%2.f", y_min), @sprintf("%2.f", y_max)])
                else
                    xticks!([])
                    yticks!([])
                end

                if alpha_mask == 1
                    plot!(NaN, NaN, background_color=:black)
                else
                    plot!(NaN, NaN, background_color="white")
                end

                plot!(
                model.acts[l][:, i][rank], 
                model.post_acts[l][:, j, i][rank], 
                color=color, 
                lw=5,
                legend=false
                )

                if sample
                    scatter!(
                        model.acts[l][:, i][rank], 
                        model.post_acts[l][:, j, i][rank], 
                        color=color, 
                        markersize=400 * scale ^ 2,
                        legend=false
                    )
                end
                
                savefig(fig, "$folder/splines/sp_$(l)_$(i)_$(j).png")

                function score2alpha(score)
                    return tanh.(γ .* score)
                end
                
                alpha = [score2alpha(score) for score in model.acts_scale] |> collect
                widths = model.widths
                A = 1.0
                y0 = 0.4
                neuron_depth = length(widths)
                min_spacing = A / max(widths)
                max_neuron = maximum(width)
                max_num_weights = maximum(width[1:end-1] .* width[2:end])
                y1 = 0.4 / max(max_num_weights, 3)

                max_num_weights = maximum(width[1:end-1] .* width[2:end])
                y1 = 0.4 / max(max_num_weights, 3)

                fig = plot(size=(10 * scale * 100, 10 * scale * (neuron_depth - 1) * y0 * 100))

                # Plot scatters and lines
                for l in 1:neuron_depth
                    n = width[l]
                    spacing = A / n
                    for i in 1:n
                        scatter!(1 / (2 * n) + (i - 1) / n, (l - 1) * y0, markersize=min_spacing^2 * 10000 * scale^2, color="black", legend=false)
                        
                        if l < neuron_depth
                            # Plot connections
                            n_next = width[l + 1]
                            N = n * n_next
                            for j in 1:n_next
                                id_ = (i - 1) * n_next + j
                                
                                symbol_mask = model.symbolic_fcns[l].mask[j, i]
                                numerical_mask = model.act_fcns[l].mask[i, j]
                                if symbol_mask == 1.0 && numerical_mask == 1.0
                                    color = "purple"
                                    alpha_mask = 1.0
                                elseif symbol_mask == 1.0 && numerical_mask == 0.0
                                    color = "red"
                                    alpha_mask = 1.0
                                elseif symbol_mask == 0.0 && numerical_mask == 1.0
                                    color = "black"
                                    alpha_mask = 1.0
                                else
                                    color = "white"
                                    alpha_mask = 0.0
                                end

                                if mask == true
                                    plot!([1 / (2 * n) + (i - 1) / n, 1 / (2 * N) + (id_ - 1) / N], [(l - 1) * y0, (l - 0.5) * y0 - y1], color=color, lw=2 * scale, alpha=alpha[l][j][i] * model.mask[l][i] * model.mask[l + 1][j], legend=false)
                                    plot!([1 / (2 * N) + (id_ - 1) / N, 1 / (2 * n_next) + (j - 1) / n_next], [(l - 0.5) * y0 + y1, l * y0], color=color, lw=2 * scale, alpha=alpha[l][j][i] * model.mask[l][i] * model.mask[l + 1][j], legend=false)
                                else
                                    plot!([1 / (2 * n) + (i - 1) / n, 1 / (2 * N) + (id_ - 1) / N], [(l - 1) * y0, (l - 0.5) * y0 - y1], color=color, lw=2 * scale, alpha=alpha[l][j][i] * alpha_mask, legend=false)
                                    plot!([1 / (2 * N) + (id_ - 1) / N, 1 / (2 * n_next) + (j - 1) / n_next], [(l - 0.5) * y0 + y1, l * y0], color=color, lw=2 * scale, alpha=alpha[l][j][i] * alpha_mask, legend=false)
                                end
                            end
                        end
                    end
                end

                xlims!(0, 1)
                ylims!(-0.1 * y0, (neuron_depth - 1 + 0.1) * y0)
                axis_off!()

                # Transformation functions
                DC_to_FC = ax.transData.transform
                FC_to_NFC = fig.transFigure.inverted().transform
                DC_to_NFC = x -> FC_to_NFC(DC_to_FC(x))

                # Plot splines
                for l in 1:(neuron_depth - 1)
                    n = width[l]
                    for i in 1:n
                        n_next = width[l + 1]
                        N = n * n_next
                        for j in 1:n_next
                            id_ = (i - 1) * n_next + j
                            im = load("$folder/sp_$(l - 1)_$(i - 1)_$(j - 1).png")
                            left = DC_to_NFC([1 / (2 * N) + (id_ - 1) / N - y1, 0])[1]
                            right = DC_to_NFC([1 / (2 * N) + (id_ - 1) / N + y1, 0])[1]
                            bottom = DC_to_NFC([0, (l - 0.5) * y0 - y1])[2]
                            up = DC_to_NFC([0, (l - 0.5) * y0 + y1])[2]
                            newax = fig.add_axes([left, bottom, right - left, up - bottom])
                            if mask == false
                                newax.imshow(im, alpha=alpha[l][j][i])
                            else
                                newax.imshow(im, alpha=alpha[l][j][i] * self.mask[l][i] * self.mask[l + 1][j])
                            end
                            newax.axis("off")
                        end
                    end
                end

                if isnothing(in_vars)
                    n = self.width[1]
                    for i in 1:n
                        annotate!(1 / (2 * n) + (i - 1) / n, -0.1, text(in_vars[i], 40 * scale, :center))
                    end
                end

                if isnothing(out_vars)
                    n = self.width[end]
                    for i in 1:n
                        annotate!(1 / (2 * n) + (i - 1) / n, y0 * (neuron_depth - 1) + 0.1, text(out_vars[i], 40 * scale, :center))
                    end
                end

                if !isnothing(title)
                    annotate!(0.5, y0 * (neuron_depth - 1) + 0.2, text(title, 40 * scale, :center))
                end

                savefig(fig, "$folder/sp_$(l)_$(i)_$(j).png")
            end
        end
    end
end

end