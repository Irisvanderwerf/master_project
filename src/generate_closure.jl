using Plots
using IncompressibleNavierStokes
const INS = IncompressibleNavierStokes
using OrdinaryDiffEq
using FFTW
using KernelAbstractions
using Printf

# implement Heuns method. 

function euler_maruyama(x, b_F, Δt, sigma, dev)
    wiener = randn(size(x)).* sqrt(Δt) |> dev
    noise_term = sigma .* wiener |> dev
    return x .+ b_F * Δt .+ noise_term |> dev
end

function generate_closure(velocity_cnn, ps_drift, _st_drift, target_label_closure_test, target_label_state_test, initial_test, num_steps, dev)
    cond_closure = Float32.(target_label_closure_test) |> dev
    cond_state = Float32.(target_label_state_test) |> dev
    images = Float32.(initial_test) |> dev
    num_test_samples = size(target_label_closure_test, 4)

    t_range = LinRange(0, 1, num_steps)
    dt = t_range[2] - t_range[1]
    for i in 1:num_steps-1
        t = t_range[i]
        t_sample = Float32.(fill(t, (1,1,1, num_test_samples))) |> dev

        drift, _st_drift = Lux.apply(velocity_cnn, (images, t_sample, cond_closure, cond_state), ps_drift, _st_drift) |> dev
        sigma_value  = sigma(t_sample, 0.1) |> dev
        images = euler_maruyama(images, drift, dt, sigma_value, dev) |> dev
    end
    return images
end

function diffusion(t_sample, ϵ)
    g = ϵ .* sqrt.((3 .- t_sample) .* (1 .- t_sample))
    return g
end

function score(t_sample, images, closure, drift)
    epsilon = 1e-10
    c = derivative_beta(t_sample) .* images .+ (beta(t_sample) .* derivative_alpha() .- derivative_beta(t_sample) .* alpha(t_sample)) .* closure 
    A = inv.(t_sample .* sigma(t_sample, 0.1) .* (derivative_beta(t_sample) .* sigma(t_sample, 0.1) - beta(t_sample) .* derivative_sigma(0.1)))
    A_inverted = 1 ./ (A .+ epsilon)
    score = A_inverted .* (beta(t_sample) .* drift .- c)
    return score
end

function generate_closure_with_tunable_diffusion(velocity_cnn, ps_drift, _st_drift, target_label_closure_test, target_label_state_test, initial_test, num_steps, dev)
    cond_closure = Float32.(target_label_closure_test) |> dev
    cond_state = Float32.(target_label_state_test) |> dev
    images = Float32.(initial_test) |> dev
    num_test_samples = size(target_label_closure_test, 4)

    t_range = LinRange(0, 1, num_steps)
    dt = t_range[2] - t_range[1]
    for i in 1:num_steps-1
        t = t_range[i]
        t_sample = Float32.(fill(t, (1,1,1, num_test_samples))) |> dev

        drift, _st_drift = Lux.apply(velocity_cnn, (images, t_sample, cond_closure, cond_state), ps_drift, _st_drift) |> dev

        g = diffusion(t_sample, 1) |> dev
        score_value = score(t_sample, images, cond_closure, drift) |> dev
        sigma_value = sigma(t_sample, 1) |> dev

        tunable_drift = drift .+ ((1/2) .* (g.^2  - sigma_value.^2) .* score_value) |> dev
        images = euler_maruyama(images, tunable_drift, dt, g, dev) |> dev
    end
    return images
end

function generate_closure_test_trajectory(velocity_cnn, ps_drift, _st_drift, num_initial_conditions, num_train_conditions, initial_test, target_label_closure_test, target_label_state_test, dev)
    num_steps = 200; 
    trajectory_to_evaluate = rand(1:(num_initial_conditions-num_train_conditions)) |> dev;
    initial_test = initial_test[:,:,:,:,trajectory_to_evaluate] |> dev;
    target_label_closure_test = target_label_closure_test[:,:,:,:,trajectory_to_evaluate] |> dev;
    target_label_state_test = target_label_state_test[:,:,:,:,trajectory_to_evaluate] |> dev;
    predicted_closure = generate_closure(velocity_cnn, ps_drift, _st_drift, target_label_closure_test, target_label_state_test, initial_test, num_steps, dev)
    return predicted_closure, trajectory_to_evaluate
end

function generate_closure_test_trajectory_tunable_drift(velocity_cnn, ps_drift, _st_drift, initial_test, target_label_closure_test, target_label_state_test, trajectory_to_evaluate, dev)
    num_steps = 200; 
    initial_test = initial_test[:,:,:,:,trajectory_to_evaluate] |> dev;
    target_label_closure_test = target_label_closure_test[:,:,:,:,trajectory_to_evaluate] |> dev;
    target_label_state_test = target_label_state_test[:,:,:,:,trajectory_to_evaluate] |> dev;
    predicted_closure = generate_closure_with_tunable_diffusion(velocity_cnn, ps_drift, _st_drift, target_label_closure_test, target_label_state_test, initial_test, num_steps, dev)
    return predicted_closure
end