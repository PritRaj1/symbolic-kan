module Plotting

export plot_kan!

using Flux, Statistics, Makie, GLMakie, FileIO, Printf, IntervalSets

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

function plot_kan!(model; folder="figures/", μ=100, γ=3, mask=false, mode="supervised", σ=1.0, tick=false, sample=false, in_vars=nothing, out_vars=nothing, title=nothing)
    """
    Plot KAN.

    Args:
    - model: KAN model.
    - folder: folder to save plots.
    - γ: controls transparaency of each activation.
    - mask: whether to prune and plot mask. true -> prune and plot mask; false -> plot all activations.
    - mode: "supervised" or "unsupervised". "supervised" -> l1 is measured by absolution value; "unsupervised" -> l1 is measured by standard deviation.
    - σ: scale of diagram.
    - tick: whether to show ticks.
    - sample: whether to sample activations.
    - in_vars: input variables.
    - out_vars: output variables.
    - title: title of plot.
    """

    # Create folder if it does not exist
    !isdir(folder) && mkdir(folder)    
    !isdir(folder * "splines/") && mkdir(folder * "splines/")

    depth = length(model.widths) - 1

    for l in 1:depth
        w_large = 2.0
        for i in 1:model.widths[l]
            for j in 1:model.widths[l+1]
                rank = sortperm(view(model.acts[l][:, i], :), rev=true)

                symbol_mask = model.symbolic_fcns[l].mask[j, i]
                numerical_mask = model.act_fcns[l].mask[i, j]

                fig = Figure(
                    size = (w_large * μ, w_large * μ),
                    ffont="Computer Modern",
                    fontsize = 20,
                    backgroundcolor = :white,
                    show_axis = false,
                    show_grid = false,
                    show_tickmarks = tick,
                    show_axis_labels = false,
                    show_legend = false,
                    show_colorbar = false,
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

    sizes = (10μ, 10μ * (neuron_depth - 1) * y0)
    fig = Figure(size=sizes,
                font="Computer Modern",
                fontsize=20, 
                backgroundcolor=:white, 
                show_axis=false, 
                show_grid=false, 
                show_tickmarks=false, 
                show_axis_labels=false, 
                show_legend=false, 
                show_colorbar=false)

    ax = Axis(fig[1, 1], aspect = DataAspect())
    x_lim = (0, 1)
    y_lim = (-0.2 * y0, (neuron_depth - 1 + 0.75) * y0)
    limits!(ax, x_lim..., y_lim...)
    hidedecorations!(ax)
    hidespines!(ax)

    color = :purple
    for l in 1:neuron_depth
        n = widths[l]
        spacing = A / n
        for i in 1:n
            scatter!(ax, [1 / (2 * n) + (i-1) / n], [(l-1) * y0], 
                        color=:black, 
                        markersize=min_spacing^2 * 700 * σ^2)
            
            n_next = (l < neuron_depth) ? widths[l + 1] : 1
            N = n * n_next

            for j in 1:n_next
                id_ = (i-1) * n_next + (j-1) 

                if l < neuron_depth - 1
                    
                    symbol_mask = model.symbolic_fcns[l].mask[j, i]
                    numerical_mask = model.act_fcns[l].mask[i, j]
                    
                    if symbol_mask > 0 && numerical_mask > 0
                        color = :purple
                        alpha_mask = 1.0
                    elseif symbol_mask > 0 && numerical_mask == 0
                        color = :red
                        alpha_mask = 1.0
                    elseif symbol_mask == 0.0 && numerical_mask > 0
                        color = :black
                        alpha_mask = 1.0
                    else
                        color = :white
                        alpha_mask = 0.0
                    end

                    if alpha_mask == 0.0
                        hidespines!(ax)
                    end

                    alpha_plot = mask ? alpha[l, j, i] * model.mask[l][i] * model.mask[l + 1][j] : alpha[l, j, i] * alpha_mask

                else
                    alpha_plot = mask ? model.act_fcns[end].mask[end] * alpha[end, j, i] : alpha[end, j, i] * alpha_mask
                    alpha_plot = l == neuron_depth ? 0.0 : alpha_plot # Remove last line
                end

                lines!(ax, [1 / (2 * N) + id_ / N, 1 / (2 * n_next) + (j-1) / n_next], 
                        [(l - 1 / 2) * y0 + y1, l * y0], 
                        color=color, 
                        linewidth=2σ, 
                        alpha=alpha_plot)

                lines!(ax, [1 / (2 * n) + (i-1) / n, 1 / (2 * N) + id_ / N], 
                        [(l-1) * y0, (l - 1 / 2) * y0 - y1], 
                        color=color, 
                        linewidth=2σ, 
                        alpha=alpha_plot)
            end
        end
    end

    # Transformation functions
    DC_to_FC = point -> Makie.apply_transform((ax.scene.transformation.transform_func[]), Point2f(point[1], point[2]))
    FC_to_NFC = point -> Makie.apply_transform(Makie.inverse_transform((ax.scene.transformation.transform_func[])), Point2f(point[1], point[2]))
    DC_to_NFC = point -> FC_to_NFC(DC_to_FC(point))

    # Plot splines
    for l in 1:neuron_depth-1
        n = widths[l]
        for i in 1:(n)
            n_next = widths[l + 1]
            N = n * n_next
            for j in 1:(n_next)
                id_ = (i-1) * n_next + (j-1)

                im = load(folder * "splines/sp_$(l)_$(i)_$(j).png")
                
                left = DC_to_NFC([1 / (2 * N) + id_ / N - y1, 0])[1] |> Float32
                right = DC_to_NFC([1 / (2 * N) + id_ / N + y1, 0])[1] |> Float32
                bottom = DC_to_NFC([0, (l - 1 / 2) * y0 - y1])[2] |> Float32
                top = DC_to_NFC([0, (l - 1 / 2) * y0 + y1])[2] |> Float32
                
                image_alpha = mask ? alpha[l, j, i] * model.mask[l][i] * model.mask[l+1][j] : alpha[l, j, i]                         
                image!(ax, left..right, bottom..top, rotr90(im), alpha=image_alpha)
            end
        end
    end

    # Add input variable labels
    if !isnothing(in_vars)
        n = widths[1]
        for (i, var) in enumerate(in_vars)
            text!(fig[1, 1], 1 / (2 * n) + (i-1) / n, -0.1*y0, 
                    text=var, align=(:center, :center), fontsize=20σ)
        end
    end
    
    # Add output variable labels
    if !isnothing(out_vars)
        n = widths[end]
        for (i, var) in enumerate(out_vars)
            text!(fig[1, 1], 1 / (2 * n) + (i-1) / n, y0 * (length(widths) - 1 + 0.25), 
                    text=var, align=(:center, :center), fontsize=20σ)
        end
    end

    # Add title
    if !isnothing(title)
        text!(fig[1, 1], 0.5, y0 * (length(widths) - 1 + 0.5), text=title, 
                align=(:center, :center), fontsize=20σ)

    end
    save(folder * "kan.png", fig)
end

end