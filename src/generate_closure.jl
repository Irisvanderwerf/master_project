using Plots
using IncompressibleNavierStokes
const INS = IncompressibleNavierStokes
using OrdinaryDiffEq
using FFTW
using KernelAbstractions
using Printf
using CUDA

function euler_maruyama(x, b_F, Δt, sigma)
    shape = size(x)
    wiener = CUDA.randn(shape...) .* sqrt(Δt)
    noise_term = sigma .* wiener
    return x .+ b_F * Δt .+ noise_term
end

function step_rk4_with_closure(u1, c0, dt, velocity_cnn, ps_drift, _st_drift, num_steps, ϵ, N_les, f_les_non_proj, setup_les, psolver_les; dev, method=:euler_maruyama)
    u = u1;
    c = c0;

    input_model_c_1 = reshape(c, N_les, N_les, 2, 1);
    input_model_u_1 = reshape(u, N_les, N_les, 2, 1);

    if method == :euler_maruyama
        closure_1_dim = generate_closure(velocity_cnn, ps_drift, _st_drift, input_model_c_1, input_model_u_1, input_model_c_1, num_steps, ϵ, dev; method=:euler_maruyama) |> dev;
    elseif method == :heuns_method
        closure_1_dim = generate_closure(velocity_cnn, ps_drift, _st_drift, input_model_c_1, input_model_u_1, input_model_c_1, num_steps, ϵ, dev; method=:heuns_method) |> dev;
    end
    closure_1 = closure_1_dim[:,:,:,1] |> dev;
    non_proj_closure_pad = pad_circular(closure_1, 1; dims=1:2)   
    proj_closure_1_pad = INS.project(non_proj_closure_pad, setup_les; psolver=psolver_les)
    proj_closure_1 = proj_closure_1_pad[2:end-1, 2:end-1, :] |> dev;

    closure_half = 1/2 .* (c .+ proj_closure_1) |> dev;

    non_proj_k1 = f_les_non_proj(u, nothing, 0.0) .+ closure_1 |> dev;
    non_proj_k1_pad = pad_circular(non_proj_k1, 1; dims=1:2) |> dev;
    proj_k1 = INS.project(non_proj_k1_pad, setup_les; psolver=psolver_les) |> dev;
    k1 = proj_k1[2:end-1, 2:end-1, :] |> dev;

    input_model_c_2 = reshape(closure_half, N_les, N_les, 2, 1);
    input_model_u_2 = reshape((u .+ ((dt ./2) .* k1)), N_les, N_les, 2, 1);

    if method == :euler_maruyama
        closure_2_dim = generate_closure(velocity_cnn, ps_drift, _st_drift, input_model_c_2, input_model_u_2, input_model_c_2, num_steps, ϵ, dev; method=:euler_maruyama) |> dev;
    elseif method == :heuns_method
        closure_2_dim = generate_closure(velocity_cnn, ps_drift, _st_drift, input_model_c_2, input_model_u_2, input_model_c_2, num_steps, ϵ, dev; method=:heuns_method) |> dev;
    end
    closure_2 = closure_2_dim[:,:,:,1] |> dev;
    non_proj_k2 = f_les_non_proj((u .+ ((dt ./2) .* k1)), nothing, 0.0) .+ closure_2 |> dev;
    non_proj_k2_pad = pad_circular(non_proj_k2, 1; dims=1:2)   
    proj_k2 = INS.project(non_proj_k2_pad, setup_les; psolver=psolver_les)
    k2 = proj_k2[2:end-1, 2:end-1, :] |> dev

    input_model_c_3 = reshape(closure_half, N_les, N_les, 2, 1);
    input_model_u_3 = reshape((u .+ ((dt ./2) .* k2)), N_les, N_les, 2, 1);

    if method == :euler_maruyama
        closure_3_dim = generate_closure(velocity_cnn, ps_drift, _st_drift, input_model_c_3, input_model_u_3, input_model_c_3, num_steps, ϵ, dev; method=:euler_maruyama) |> dev;
    elseif method == :heuns_method
        closure_3_dim = generate_closure(velocity_cnn, ps_drift, _st_drift, input_model_c_3, input_model_u_3, input_model_c_3, num_steps, ϵ, dev; method=:heuns_method) |> dev;
    end
    closure_3 = closure_3_dim[:,:,:,1] |> dev;
    non_proj_k3 = f_les_non_proj((u .+ ((dt ./2) .* k2)), nothing, 0.0) .+ closure_3 |> dev;
    non_proj_k3_pad = pad_circular(non_proj_k3, 1; dims=1:2)   
    proj_k3 = INS.project(non_proj_k3_pad, setup_les; psolver=psolver_les)
    k3 = proj_k3[2:end-1, 2:end-1, :] |> dev;

    input_model_c_4 = reshape(closure_1, N_les, N_les, 2, 1);
    input_model_u_4 = reshape((u .+ (dt .* k3)), N_les, N_les, 2, 1);

    if method == :euler_maruyama
        closure_4_dim = generate_closure(velocity_cnn, ps_drift, _st_drift, input_model_c_4, input_model_u_4, input_model_c_4, num_steps, ϵ, dev; method=:euler_maruyama) |> dev;
    elseif method == :heuns_method
        closure_4_dim = generate_closure(velocity_cnn, ps_drift, _st_drift, input_model_c_4, input_model_u_4, input_model_c_4, num_steps, ϵ, dev; method=:heuns_method) |> dev;
    end
    closure_4 = closure_4_dim[:,:,:,1] |> dev;
    non_proj_k4 = f_les_non_proj((u .+ (dt .* k3)), nothing, 0.0) .+ closure_4 |> dev;
    non_proj_k4_pad = pad_circular(non_proj_k4, 1; dims=1:2)   
    proj_k4 = INS.project(non_proj_k4_pad, setup_les; psolver=psolver_les)
    k4 = proj_k4[2:end-1, 2:end-1, :] |> dev;

    unew = u .+ (dt ./ 6) .* (k1 .+ 2k2 .+ 2k3 .+ k4) |> dev;
    return unew, proj_closure_1
end

function step_rk4_with_closure_deterministic(u0, dt, velocity_cnn_det, ps_deterministic, _st_deterministic, N_les, f_les_non_proj, setup_les, psolver_les; dev)
    u = u0

    input_model_u_1 = reshape(u, N_les, N_les, 2, 1);
    closure_1_dim, _ = Lux.apply(velocity_cnn_det, (input_model_u_1), ps_deterministic, _st_deterministic) |> dev;
    closure_1 = closure_1_dim[:,:,:,1] |> dev;

    non_proj_k1 = f_les_non_proj(u, nothing, 0.0) .+ closure_1 |> dev;
    non_proj_k1_pad = pad_circular(non_proj_k1, 1; dims=1:2) |> dev;
    proj_k1 = INS.project(non_proj_k1_pad, setup_les; psolver=psolver_les) |> dev;
    k1 = proj_k1[2:end-1, 2:end-1, :] |> dev;

    input_model_u_2 = reshape((u .+ ((dt ./2) .* k1)), N_les, N_les, 2, 1);
    closure_2_dim, _ = Lux.apply(velocity_cnn_det, (input_model_u_2), ps_deterministic, _st_deterministic) |> dev;
    closure_2 = closure_2_dim[:,:,:,1] |> dev;
    non_proj_k2 = f_les_non_proj((u .+ ((dt ./2) .* k1)), nothing, 0.0) .+ closure_2 |> dev;
    non_proj_k2_pad = pad_circular(non_proj_k2, 1; dims=1:2)   
    proj_k2 = INS.project(non_proj_k2_pad, setup_les; psolver=psolver_les)
    k2 = proj_k2[2:end-1, 2:end-1, :] |> dev

    input_model_u_3 = reshape((u .+ ((dt ./2) .* k2)), N_les, N_les, 2, 1);
    closure_3_dim, _ = Lux.apply(velocity_cnn_det, (input_model_u_3), ps_deterministic, _st_deterministic) |> dev;
    closure_3 = closure_3_dim[:,:,:,1] |> dev;
    non_proj_k3 = f_les_non_proj((u .+ ((dt ./2) .* k2)), nothing, 0.0) .+ closure_3 |> dev;
    non_proj_k3_pad = pad_circular(non_proj_k3, 1; dims=1:2)   
    proj_k3 = INS.project(non_proj_k3_pad, setup_les; psolver=psolver_les)
    k3 = proj_k3[2:end-1, 2:end-1, :] |> dev;

    input_model_u_4 = reshape((u .+ (dt .* k3)), N_les, N_les, 2, 1);
    closure_4_dim, _ = Lux.apply(velocity_cnn_det, (input_model_u_4), ps_deterministic, _st_deterministic) |> dev;
    closure_4 = closure_4_dim[:,:,:,1] |> dev;
    non_proj_k4 = f_les_non_proj((u .+ (dt .* k3)), nothing, 0.0) .+ closure_4 |> dev;
    non_proj_k4_pad = pad_circular(non_proj_k4, 1; dims=1:2)   
    proj_k4 = INS.project(non_proj_k4_pad, setup_les; psolver=psolver_les)
    k4 = proj_k4[2:end-1, 2:end-1, :] |> dev;

    unew = u .+ (dt ./ 6) .* (k1 .+ 2k2 .+ 2k3 .+ k4) |> dev;
    return unew, closure_1
end

function heuns_method(velocity_cnn, ps_drift, _st_drift, ϵ, images, drift_euler, sigma_euler, dt, t_sample_heun, cond_state, cond_closure; dev) 
    prediction_step = euler_maruyama(images, drift_euler, dt, sigma_euler) |> dev
    drift_heun, _st_drift = Lux.apply(velocity_cnn, (prediction_step, t_sample_heun, cond_closure, cond_state), ps_drift, _st_drift) |> dev
    sigma_heun = sigma(t_sample_heun, ϵ) |> dev 
    images = euler_maruyama(images, ((1/2) .* (drift_euler .+ drift_heun)), dt, ((1/2) .* (sigma_euler .+ sigma_heun))) 
    return images
end

function generate_closure_SI(target_closure, initial_test, num_steps, ϵ, dev; method=:euler_maruyama)
    cond_closure = Float32.(target_closure) |> dev
    initial_closure = Float32.(initial_test) |> dev
    images = Float32.(initial_test) |> dev
    num_test_samples = size(target_closure, 4)
    t_range = LinRange(0, 1, num_steps)
    dt = t_range[2] - t_range[1]
    t_sample_euler = CUDA.zeros(Float32, 1, 1, 1, num_test_samples)
    t_sample_heun = CUDA.zeros(Float32, 1, 1, 1, num_test_samples)
    for i in 1:num_steps-1
        t_sample_euler .= t_range[i] |> dev
        t_sample_heun .= t_range[i+1]
        W_euler = sqrt(t_range[i]) .* randn(size(cond_closure)) |> dev
        W_heun = sqrt(t_range[i+1]) .* randn(size(cond_closure)) |> dev
        drift_euler = time_derivative_stochastic_interpolant(initial_closure, cond_closure, W_euler, t_sample_euler, ϵ) |> dev
        sigma_value_euler = sigma(t_sample_euler, ϵ) |> dev
        if method == :euler_maruyama
            images = euler_maruyama(images, drift_euler, dt, sigma_value_euler) |> dev  
        elseif method == :heuns_method
            drift_heun = time_derivative_stochastic_interpolant(initial_closure, cond_closure, W_heun, t_sample_heun, ϵ) |> dev
            sigma_heun = sigma(t_sample_heun, ϵ) |> dev 
            images = euler_maruyama(images, ((1/2) .* (drift_euler .+ drift_heun)), dt, ((1/2) .* (sigma_value_euler .+ sigma_heun))) 
        end
    end
    return images
end


function generate_closure(velocity_cnn, ps_drift, _st_drift, target_label_closure_test, target_label_state_test, initial_test, num_steps, ϵ, dev; method=:euler_maruyama)
    cond_closure = Float32.(target_label_closure_test) |> dev
    cond_state = Float32.(target_label_state_test) |> dev
    images = Float32.(initial_test) |> dev
    num_test_samples = size(target_label_closure_test, 4)

    t_range = LinRange(0, 1, num_steps)
    dt = t_range[2] - t_range[1]
    t_sample_euler = CUDA.zeros(Float32, 1, 1, 1, num_test_samples)
    t_sample_heun = CUDA.zeros(Float32, 1, 1, 1, num_test_samples)

    for i in 1:num_steps-1
        t_sample_euler .= t_range[i]
        t_sample_heun .= t_range[i+1]

        drift_euler, _st_drift = Lux.apply(velocity_cnn, (images, t_sample_euler, cond_closure, cond_state), ps_drift, _st_drift)
        sigma_value_euler = sigma(t_sample_euler, ϵ) |> dev
        if method == :euler_maruyama
            images = euler_maruyama(images, drift_euler, dt, sigma_value_euler) |> dev  
        elseif method == :heuns_method
            images = heuns_method(velocity_cnn, ps_drift, _st_drift, ϵ, images, drift_euler, sigma_value_euler, dt, t_sample_heun, cond_state, cond_closure; dev) |> dev
        end
    end
    return images
end

function diffusion(t_sample, ϵ)
    g = ϵ .* sqrt.((3 .- t_sample) .* (1 .- t_sample))
    return g
end

function score(t_sample, images, closure, drift, ϵ)
    epsilon = 1e-10
    c = derivative_beta(t_sample) .* images .+ (beta(t_sample) .* derivative_alpha() .- derivative_beta(t_sample) .* alpha(t_sample)) .* closure 
    A = t_sample .* sigma(t_sample, ϵ) .* (derivative_beta(t_sample) .* sigma(t_sample, ϵ) - beta(t_sample) .* derivative_sigma(ϵ))
    A_inverted = 1 ./ (A .+ epsilon)
    score = A_inverted .* (beta(t_sample) .* drift .- c)
    return score
end

