using Random
using Distributions

# Define gamma(t) as given
function gamma(t)
    return sqrt(2 * t * (1 - t))
end

# Define epsilon(t) as a time-dependent diffusion coefficient
function epsilon(t)
    return t * (1 - t)
end

# Euler-Maruyama method for SDE
function euler_maruyama(x, b_F, epsilon_t, Δt)
    noise_term = sqrt(2 * epsilon_t) * randn(size(x)) * sqrt(Δt)
    return x + b_F * Δt + noise_term
end

# Function to compute the score safely (handling t=0 and t=1)
function compute_score(t, denoiser)
    if t == 0 || t == 1
        return 0  # Set score to zero at boundaries
    else
        return -denoiser ./ gamma(t)  # Regularized score for t in (0, 1)
    end
end

# Function to generate samples using the drift and score
function generate_digit(gaussian_image, label, batch_size, Δt, ps_drift, st_drift, ps_denoiser, st_denoiser, velocity_cnn, dev)
    # Initialize time
    t = 0
    # Set label to the right size
    label = fill(label, 32, 32, 1, batch_size) |> dev
    
    while t < 1
        # Reshape t_sample to match the right size
        t_sample = Float32.(fill(t, (1, 1, 1, batch_size)))
        # Compute the score (denoiser-based term)
        denoiser, st_denoiser = Lux.apply(velocity_cnn, (gaussian_image, t_sample, label), ps_denoiser, st_denoiser)
        score = compute_score(t, denoiser)  # Construct score based on the denoiser
        
        # Compute the drift term
        drift, st_drift = Lux.apply(velocity_cnn, (gaussian_image, t_sample, label), ps_drift, st_drift)
        
        # Construct the forward drift b_F(t, x)
        b_F = drift .+ epsilon(t) .* score
        
        # Propagate the samples using the Euler-Maruyama method
        gaussian_image = euler_maruyama(gaussian_image, b_F, epsilon(t), Δt)
        
        # Update time
        t += Δt
    end
    
    # Return the generated samples
    return gaussian_image
end

# plot the generated images
function plot_generated_digits(images, num_images_to_show)
    num_cols = ceil(Int, sqrt(num_images_to_show))
    num_rows = ceil(Int, num_images_to_show / num_cols)

    p = plot(layout=(num_rows, num_cols), size=(800, 800))
    
    for i in 1:num_images_to_show
        img = reshape(images[:, :, 1, i], (32, 32))  # Reshape to (28, 28) - changed to (32,32)
        heatmap!(img, color=:grays, axis=false, legend=false, subplot=i, title="Generated Image")
    end
    
    display(p)
end

