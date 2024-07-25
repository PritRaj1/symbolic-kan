module Plotting

export plot_kan!

using Flux, Statistics, Makie, GLMakie, FileIO, Printf, IntervalSets

include("../architecture/kan_model.jl")
using .KolmogorovArnoldNets: prune

function get_range(model, l, i, j; verbose=true)
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

function format_tick!(ax; x_min, x_max, y_min, y_max)
    """
    Format the tick labels of the axis.
    """
    ax.xticksmirrored = true
    ax.yticksmirrored = true
    ax.xlabelsize = 50
    ax.ylabelsize = 50
    ax.xlabelpadding = -15
    ax.ylabelpadding = -22
    ax.xticks = ([x_min, x_max], ["%2.f" % x_min, "%2.f" % x_max])
    ax.yticks = ([y_min, y_max], ["%2.f" % y_min, "%2.f" % y_max])
    format_ticks(x) = @sprintf("%.2f", x)
    ax.xtickformat = format_ticks
    ax.ytickformat = format_ticks
end

function plot_kan!(model; folder="figures/", γ=3, prune_and_mask=false, mode="supervised", σ=0.5, tick=false, sample=false, in_vars=nothing, out_vars=nothing, title=nothing)
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
        w_large = 500
        for i in 1:model.widths[l]
            for j in 1:model.widths[l+1]
                rank = sortperm(view(model.acts[l][:, i], :), rev=true)

                symbol_mask = model.symbolic_fcns[l].mask[j, i]
                numerical_mask = model.act_fcns[l].mask[i, j]

                fig = Figure(
                    size = (w_large, w_large),
                    font = "Arial",
                    fontsize = 20,
                    backgroundcolor = :white,
                    show_axis = false,
                    show_grid = false,
                    show_tickmarks = tick,
                    show_axis_labels = false,
                    show_legend = false,
                    show_colorbar = false,
                    scale_plot = true,
                    scale_plot_by = σ
                )

                ax = Axis(fig[1, 1])

                if tick
                    x_min, x_max, y_min, y_max = get_range(model, l, i, j; verbose=false)
                    format_tick!(ax, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
                else
                    hidedecorations!(ax)
                end

                # Determine color and alpha_mask based on conditions
                if symbol_mask == 1.0 && numerical_mask == 1.0
                    color = :purple
                    alpha_mask = 1
                elseif symbol_mask == 1.0 && numerical_mask == 0.0
                    color = :red
                    alpha_mask = 1
                elseif symbol_mask == 0.0 && numerical_mask == 1.0
                    color = :black
                    alpha_mask = 1
                else
                    color = :white
                    alpha_mask = 0
                end

                alpha_mask == 0 && hidespines!(ax)

                acts_data = model.acts[l][:, i][rank]
                spline_data = model.post_acts[l][:, j, i][rank]
                lines!(ax, acts_data, spline_data, color=color, linewidth=5)

                if sample
                    scatter!(ax, acts_data, spline_data, color=color, markersize=400 * σ^2)
                end
                
                save("$folder/splines/sp_$(l)_$(i)_$(j).png", fig)
            end
        end
    end

    function score2alpha(score)
        return tanh(γ * score)
    end
    
    alpha = score2alpha.(model.act_scale)
    widths = model.widths
    A = 1.0
    y0 = 0.4
    neuron_depth = length(widths)
    min_spacing = A / max(widths..., 5)
    max_neuron = max(widths...)
    max_num_weights = max((widths[1:end-1] .* widths[2:end])...)
    y1 = 0.4 / max(max_num_weights..., 3)

    max_num_weights = max((widths[1:end-1] .* widths[2:end])...)
    y1 = 0.4 / max(max_num_weights..., 3)

    fig = Figure(size=(2500, 2500 * (neuron_depth - 1) * y0), 
                font="Arial", 
                fontsize=20, 
                backgroundcolor=:white, 
                show_axis=false, 
                show_grid=false, 
                show_tickmarks=false, 
                show_axis_labels=false, 
                show_legend=false, 
                show_colorbar=false, 
                scale_plot=true, 
                scale_plot_by=σ)

    ax = Axis(fig[1, 1])

    for l in 1:neuron_depth
        n = widths[l]
        spacing = A / n
        for i in 1:n
            scatter!(ax, [1 / (2 * n) + i / n], [l * y0], 
                        color=:black, 
                        markersize=min_spacing^2 * 10000 * σ^2)

            if l < neuron_depth-1
                n_next = widths[l + 1]
                N = n * n_next
                for j in 1:n_next
                    id_ = i * n_next + j

                    symbol_mask = model.symbolic_fcns[l].mask[j, i]
                    numerical_mask = model.act_fcns[l].mask[i, j]
                    
                    if symbol_mask == 1.0 && numerical_mask == 1.0
                        color = :purple
                        alpha_mask = 1.0
                    elseif symbol_mask == 1.0 && numerical_mask == 0.0
                        color = :red
                        alpha_mask = 1.0
                    elseif symbol_mask == 0.0 && numerical_mask == 1.0
                        color = :black
                        alpha_mask = 1.0
                    else
                        color = :white
                        alpha_mask = 0.0
                    end

                    if alpha_mask == 0.0
                        hidespines!(ax)
                    end

                    if prune_and_mask
                        lines!(ax, [1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N], 
                                [l * y0, (l + 1 / 2) * y0 - y1], 
                                color=color, 
                                linewidth=2σ, 
                                alpha=alpha[l, j, i] * model.mask[l][i] * model.mask[l + 1][j])
                        lines!(ax, [1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next], 
                                [(l + 1 / 2) * y0 + y1, (l + 1) * y0], 
                                color=color, 
                                linewidth=2σ, 
                                alpha=alpha[l, j, i] * model.mask[l][i] * model.mask[l + 1][j])
                    else
                        lines!(ax, [1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N], 
                                [l * y0, (l + 1 / 2) * y0 - y1], 
                                color=color, 
                                linewidth=2σ, 
                                alpha=alpha[l, j, i] * alpha_mask)
                        lines!(ax, [1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next], 
                                [(l + 1 / 2) * y0 + y1, (l + 1) * y0], 
                                color=color, 
                                linewidth=2 * σ, 
                                alpha=alpha[l, j, i] * alpha_mask)
                    end
                end
            end
        end

        x_lim =(0, 1)
        y_lim = (-0.1 * y0, (neuron_depth - 1 + 0.1) * y0)
        limits!(ax, x_lim..., y_lim...)
    end

    # Transformation functions
    DC_to_FC = point -> Makie.apply_transform((ax.scene.transformation.transform_func[]), Point2f(point[1], point[2]))
    FC_to_NFC = point -> Makie.apply_transform(Makie.inverse_transform((ax.scene.transformation.transform_func[])), Point2f(point[1], point[2]))
    DC_to_NFC = point -> FC_to_NFC(DC_to_FC(point))

    # Plot splines
    for l in 1:(neuron_depth-1)
        n = widths[l]
        for i in 1:(n)
            n_next = widths[l + 1]
            N = n * n_next
            for j in 1:(n_next)
                id_ = i * n_next + j
                im = load(folder * "splines/sp_$(l)_$(i)_$(j).png")
                
                left = DC_to_NFC([1 / (2 * n) + (i-1) / n, (l-1) * y0])[1] |> Float32
                right = DC_to_NFC([1 / (2 * n) + i / n, (l-1) * y0])[1] |> Float32
                bottom = DC_to_NFC([1 / (2 * N) + (id_-1) / N, (l-1 + 1 / 2) * y0 - y1])[2] |> Float32
                top = DC_to_NFC([1 / (2 * N) + id_ / N, (l-1 + 1 / 2) * y0 + y1])[2] |> Float32
                new_axs = Axis(fig[1, 1], limits=(left, right, bottom, top))

                image_alpha = prune_and_mask ? alpha[l, i, j] * model.act_fcns[l].mask[i] * model.act_fcns[l+1].mask[j] : alpha[l, i, j]                            
                image!(new_axs, im, colormap=:grays, alpha=image_alpha)
                limits!(new_axs, left, right, bottom, top)
            end
        end
    end
    
    # Add input variable labels
    if !isnothing(in_vars)
        n = widths[1]
        for (i, var) in enumerate(in_vars)
            text!(fig[1, 1], 1 / (2 * n) + (i-1) / n, -0.1 * y0, 
                    text=var, align=(:center, :center), fontsize=40σ)
        end
    end
    
    # Add output variable labels
    if !isnothing(out_vars)
        n = widths[end]
        for (i, var) in enumerate(out_vars)
            text!(fig[1, 1], 1 / (2 * n) + (i-1) / n, y0 * (length(widths) - 1 + 0.1), 
                    text=var, align=(:center, :center), fontsize=40σ)
        end
    end

    # Add title
    if !isnothing(title)
        text!(fig[1, 1], 0.5, y0 * (length(widths) - 1 + 0.2), text=title, 
                align=(:center, :center), fontsize=40σ)

    end
    save(folder * "kan.png", fig)
end

end