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
    noise_term = sqrt(2 * epsilon_t) * randn(size(x)) * sqrt(Δt) |> gpu_device()
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

# Function to generate samples using the drift and score
function generate_closure(u_bar, u_bar_init, batch_size, Δt, ps_drift, st_drift, ps_denoiser, st_denoiser, velocity_cnn, dev, is_gaussian)
    t = 0
    u_bar = u_bar |> dev
    u_bar_init = u_bar_init |> dev
    if !is_gaussian
        while t < 1
            # Reshape t_sample to match the right size
            t_sample = Float32.(fill(t, (1, 1, 1, batch_size))) |> dev
            # Compute the score (denoiser-based term)
            denoiser, st_denoiser = Lux.apply(velocity_cnn, (u_bar, t_sample, u_bar_init), ps_denoiser, st_denoiser) |> dev
            score = compute_score_denoiser(t, denoiser) |> dev # Construct score based on the denoiser
        
            # Compute the drift term
            drift, st_drift = Lux.apply(velocity_cnn, (u_bar, t_sample, u_bar_init), ps_drift, st_drift) |> dev
        
            # Construct the forward drift b_F(t, x)
            b_F = drift .+ epsilon(t) .* score |> dev
        
            # Propagate the samples using the Euler-Maruyama method
            u_bar = euler_maruyama(u_bar, b_F, epsilon(t), Δt) |> dev
        
            # Update time
            t += Δt
        end
    else
        while t < 1
            # Reshape t_sample to match the right size
            t_sample = Float32.(fill(t, (1, 1, 1, batch_size))) |> dev
            # Compute the drift term
            drift, st_drift = Lux.apply(velocity_cnn, (u_bar, t_sample, u_bar_init), ps_drift, st_drift) |> dev
            # Compute the score
            score = compute_score_velocity(t, drift, u_bar) |> dev
            # compute the term for dt 
            b_F = drift .+ score |> dev
            # Propagate the samples using the Euler-Maruyama method
            u_bar = euler_maruyama(u_bar, b_F, epsilon(t), Δt) |> dev
            # Update time
            t += Δt
        end
    end
    # Return the generated samples
    println("Finished computing the closure")
    return u_bar
end

