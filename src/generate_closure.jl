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

# function step_rk4_with_closure(u0, c0, dt, velocity_cnn, ps_drift, _st_drift, num_steps, ϵ, N_les, Re, f_les, setup_les, psolver_les; dev)
#     u = u0
#     c = c0

#     k1_1 = f_les(u, nothing, 0.0) |> dev
#     input_model_c_1 = reshape(c, N_les, N_les, 2, 1);
#     input_model_u_1 = reshape(u, N_les, N_les, 2, 1);
#     pred_closure_1 = generate_closure(velocity_cnn, ps_drift, _st_drift, input_model_c_1, input_model_u_1, input_model_c_1, num_steps, ϵ, N_les, Re, dev; method=:euler_maruyama, time_step=:FE) |> dev
#     input_project_1 = pad_circular(pred_closure_1[:,:,:,1], 1; dims=1:2)   
#     output_project_1 = INS.project(input_project_1, setup_les; psolver=psolver_les)
#     k1_2 = output_project_1[2:end-1, 2:end-1, :] |> dev
#     k1 = k1_1 .+ k1_2 |> dev;

#     k2_1 = f_les((u .+ (dt .* (k1 ./ 2))), nothing, 0.0) |> dev
#     input_model_c_2 = reshape((c .+ (dt .* (k1_2 ./ 2))), N_les, N_les, 2, 1);
#     input_model_u_2 = reshape((u .+ (dt .* (k1 ./ 2))), N_les, N_les, 2, 1);
#     pred_closure_2 = generate_closure(velocity_cnn, ps_drift, _st_drift, input_model_c_2, input_model_u_2, input_model_c_2, num_steps, ϵ, N_les, Re, dev; method=:euler_maruyama, time_step=:FE) |> dev
#     input_project_2 = pad_circular(pred_closure_2[:,:,:,1], 1; dims=1:2)
#     output_project_2 = INS.project(input_project_2, setup_les; psolver=psolver_les)
#     k2_2 = output_project_2[2:end-1, 2:end-1, :] |> dev
#     k2 = k2_1 .+ k2_2 |> dev;

#     k3_1 = f_les((u .+ (dt .* (k2 ./ 2))), nothing, 0.0) |> dev
#     input_model_c_3 = reshape((c .+ (dt .* (k2_2 ./ 2))), N_les, N_les, 2, 1);
#     input_model_u_3 = reshape((u .+ (dt .* (k2 ./ 2))), N_les, N_les, 2, 1);
#     pred_closure_3 = generate_closure(velocity_cnn, ps_drift, _st_drift, input_model_c_3, input_model_u_3, input_model_c_3, num_steps, ϵ, N_les, Re, dev; method=:euler_maruyama, time_step=:FE) |> dev
#     input_project_3 = pad_circular(pred_closure_3[:,:,:,1], 1; dims=1:2)
#     output_project_3 = INS.project(input_project_3, setup_les; psolver=psolver_les)
#     k3_2 = output_project_3[2:end-1, 2:end-1,:] |> dev
#     k3 = k3_1 .+ k3_2 |> dev;

#     k4_1 = f_les((u .+ (dt .* k3)), nothing, 0.0) |> dev
#     input_model_c_4 = reshape((c .+ (dt .* k3_2)), N_les, N_les, 2, 1);
#     input_model_u_4 = reshape((u .+ (dt .* k3)), N_les, N_les, 2, 1);
#     pred_closure_4 = generate_closure(velocity_cnn, ps_drift, _st_drift, input_model_c_4, input_model_u_4, input_model_c_4, num_steps, ϵ, N_les, Re, dev; method=:euler_maruyama, time_step=:FE) |> dev
#     input_project_4 = pad_circular(pred_closure_4[:,:,:,1], 1; dims=1:2)
#     output_project_4 = INS.project(input_project_4, setup_les; psolver=psolver_les)
#     k4_2 = output_project_4[2:end-1, 2:end-1,:] |> dev
#     k4 = k4_1 .+ k4_2 |> dev;

#     unew = u .+ (dt ./ 6) .* (k1 .+ 2k2 .+ 2k3 .+ k4) |> dev;
#     return unew
# end

function heuns_method(velocity_cnn, ps_drift, _st_drift, ϵ, images, tunable_drift_euler, dt, g_euler, t_sample_heun, cond_state, cond_closure; dev)
    prediction_step = euler_maruyama(images, tunable_drift_euler, dt, g_euler) |> dev
    g_heun = diffusion(t_sample_heun, ϵ) |> dev
    drift_heun, _st_drift = Lux.apply(velocity_cnn, (prediction_step, t_sample_heun, cond_closure, cond_state), ps_drift, _st_drift) |> dev
    sigma_heun = sigma(t_sample_heun, ϵ) |> dev 
    score_heun = score(t_sample_heun, prediction_step, prediction_step, drift_heun, ϵ) |> dev
    tunable_drift_heun = drift_heun .+ ((1/2) .* (g_heun.^2 .- sigma_heun.^2) .* score_heun) |> dev
    images = euler_maruyama(images, ((1/2) .* (tunable_drift_euler .+ tunable_drift_heun)), dt, ((1/2) .* (g_euler .+ g_heun)))    
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

        g_euler = diffusion(t_sample_euler, ϵ) |> dev
        score_value_euler = score(t_sample_euler, images, cond_closure, drift_euler, ϵ) |> dev
        sigma_value_euler = sigma(t_sample_euler, ϵ) |> dev

        tunable_drift_euler = drift_euler .+ ((1/2) .* (g_euler.^2  - sigma_value_euler.^2) .* score_value_euler) |> dev

        if method == :euler_maruyama
            images = euler_maruyama(images, tunable_drift_euler, dt, g_euler) |> dev
        elseif method == :heuns_method
            images = heuns_method(velocity_cnn, ps_drift, _st_drift, ϵ, images, tunable_drift_euler, dt, g_euler, t_sample_heun, cond_state, cond_closure; dev) |> dev
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

