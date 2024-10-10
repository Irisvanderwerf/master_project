using Random
# using Distributions

# Define necessary information of the stochastic interpolant
function gamma(t)
    return sqrt(2 * t * (1 - t))
end

function alpha(t)
    return cos.(π/2 .* t)
end

function beta(t)
    return sin.(π/2 .* t)
end

function derivative_alpha(t)
    return -π/2 .* sin.(π/2 .* t)
end

function derivative_beta(t)
    return π/2 .* cos.(π/2 .* t)
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

# Function to compute the score safely 
function compute_score_denoiser(t, denoiser)
    if t == 0 
        return 0  # Set score to zero at boundaries
    else
        return -denoiser ./ gamma(t)  # Regularized score for t in (0, 1)
    end
end

# Function to compute the score safely expressed in terms of velocity
function compute_score_velocity(t, drift, gaussian_image)
    if t == 0
        return 0
    else
        return (epsilon(t)./(beta(t) .* ((derivative_alpha(t)./alpha(t)) .* beta(t) .- derivative_beta(t)))) .* (drift .- ((derivative_alpha(t) ./ alpha(t)) .* gaussian_image))
    end
end

# Function to generate samples using the drift and score
function generate_digit(gaussian_image, label, batch_size, Δt, ps_drift, st_drift, ps_denoiser, st_denoiser, velocity_cnn, dev, is_gaussian)
    # Initialize time
    t = 0
    # Set label to the right size
    label = (1/10) .* fill(label, 32, 32, 1, batch_size) |> dev
    if !is_gaussian
        while t < 1
            # Reshape t_sample to match the right size
            t_sample = Float32.(fill(t, (1, 1, 1, batch_size)))
            # Compute the score (denoiser-based term)
            denoiser, st_denoiser = Lux.apply(velocity_cnn, (gaussian_image, t_sample, label), ps_denoiser, st_denoiser)
            score = compute_score_denoiser(t, denoiser)  # Construct score based on the denoiser
        
            # Compute the drift term
            drift, st_drift = Lux.apply(velocity_cnn, (gaussian_image, t_sample, label), ps_drift, st_drift)
        
            # Construct the forward drift b_F(t, x)
            b_F = drift .+ epsilon(t) .* score
        
            # Propagate the samples using the Euler-Maruyama method
            gaussian_image = euler_maruyama(gaussian_image, b_F, epsilon(t), Δt)
        
            # Update time
            t += Δt
        end
    else
        while t < 1
            # Reshape t_sample to match the right size
            t_sample = Float32.(fill(t, (1, 1, 1, batch_size)))
            # Compute the drift term
            drift, st_drift = Lux.apply(velocity_cnn, (gaussian_image, t_sample, label), ps_drift, st_drift)
            # Compute the score
            score = compute_score_velocity(t, drift, gaussian_image)
            # compute the term for dt 
            b_F = drift .+ score
            # Propagate the samples using the Euler-Maruyama method
            gaussian_image = euler_maruyama(gaussian_image, b_F, epsilon(t), Δt)
            # Update time
            t += Δt
        end
        # Return the generated samples
        return gaussian_image
    end
end

