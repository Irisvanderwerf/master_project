using Plots
using IncompressibleNavierStokes
const INS = IncompressibleNavierStokes
using OrdinaryDiffEq
using FFTW
using KernelAbstractions
using Printf

# Forward Euler method
function forward_euler(velocity_cnn, ps, st, images, cond, t, dt, num_test_samples, dev)
    # Reshape t_sample to match the batch size
    t_sample = Float32.(fill(t, (1, 1, 1, num_test_samples))) |> dev

    # Predict the velocity field using the neural network
    velocity, st = Lux.apply(velocity_cnn, (images, t_sample, cond), ps, st) |> dev

    # Update the images based on the velocity field
    updated_images = images .+ dt .* velocity |> dev

    return updated_images, st
end

function runge_kutta_4(velocity_cnn, ps, st, images, cond, t, dt, num_test_samples, dev)
    # Reshape t_sample to match the batch size
    t_sample = Float32.(fill(t, (1, 1, 1, num_test_samples))) |> dev
    
    # Predict the velocity field using the neural network
    velocity, st = Lux.apply(velocity_cnn, (images, t_sample, cond), ps, st)
    k1 = dt .* velocity

    t_sample_next = Float32.(fill(t + dt/2, (1, 1, 1, num_test_samples))) |> dev
    velocity_k2, st = Lux.apply(velocity_cnn, (images .+ k1 ./ 2, t_sample_next, cond), ps, st)
    k2 = dt .* velocity_k2

    t_sample_next = Float32.(fill(t + dt/2, (1, 1, 1, num_test_samples))) |> dev
    velocity_k3, st = Lux.apply(velocity_cnn, (images .+ k2 ./ 2, t_sample_next, cond), ps, st)
    k3 = dt .* velocity_k3

    t_sample_next = Float32.(fill(t + dt, (1, 1, 1, num_test_samples))) |> dev
    velocity_k4, st = Lux.apply(velocity_cnn, (images .+ k3, t_sample_next, cond), ps, st)
    k4 = dt .* velocity_k4

    # Update the images based on the RK4 method
    updated_images = images .+ (k1 .+ 2k2 .+ 2k3 .+ k4) ./ 6
    
    return updated_images, st
end

# # Main function to generate digits
# function generate_digit(velocity_cnn, ps, st, initial_gaussian_image, label, num_steps, batch_size, dev; method=:rk4)
#     images = Float32.(initial_gaussian_image)  # Ensure initial images are Float32
#     label = fill(label, 32, 32, 1, batch_size) # Set to proper size for the conditioning 
    
#     t_range = LinRange(0, 1, num_steps)
#     dt = t_range[2] - t_range[1]

#     # Simulate forward evolution over time from t = 0 to t = 1
#     for i in 1:num_steps-1
#         # Compute the current time t in the interval [0, 1]
#         t = t_range[i]
        
#         # Choose the integration method
#         if method == :euler
#             images, st = forward_euler(velocity_cnn, ps, st, images, label, t, dt, batch_size, dev)
#         elseif method == :rk4
#             images, st = runge_kutta_4(velocity_cnn, ps, st, images, label, t, dt, batch_size, dev)
#         else
#             error("Unknown method: $method. Use :euler or :rk4.")
#         end
#     end

#     # Maybe add clamping 
#     return images # clamp.(images, 0.0, 1.0)
# end

# ### Plot multiple generated images ###
# function plot_generated_digits(images, num_images_to_show)
#     num_cols = ceil(Int, sqrt(num_images_to_show))
#     num_rows = ceil(Int, num_images_to_show / num_cols)

#     p = plot(layout=(num_rows, num_cols), size=(800, 800))
    
#     for i in 1:num_images_to_show
#         img = reshape(images[:, :, 1, i], (32, 32))  # Reshape to (28, 28) - changed to (32,32)
#         heatmap!(img, color=:grays, axis=false, legend=false, subplot=i, title="Generated Image")
#     end
    
#     display(p)
# end

# generate the closure 
function generate_closure(velocity_cnn, ps_drift, st_drift, ps_denoiser, st_denoiser, cond, v_test, num_steps, is_gaussian, dev; time_method=:rk4, method=:ODE)
    if method == :ODE
        cond = Float32.(cond) |> dev
        images = randn(size(cond)) |> dev
        num_test_samples = size(cond,4)

        t_range = LinRange(0, 1, num_steps) 
        dt = t_range[2] - t_range[1]
        # Simulate forward evolution over time from t = 0 to t = 1
        for i in 1:num_steps-1
            # Compute the current time t in the interval [0, 1]
            t = t_range[i]
            images = Float32.(images) |> dev
        
            # Choose the integration method
            if time_method == :euler
                images, st = forward_euler(velocity_cnn, ps_drift, st_drift, images, cond, t, dt, num_test_samples, dev) |> dev
            elseif time_method == :rk4
                images, st = runge_kutta_4(velocity_cnn, ps_drift, st_drift, images, cond, t, dt, num_test_samples, dev) |> dev
            else
                error("Unknown method: $time_method. Use :euler or :rk4.")
            end
        end
        # Return the generated samples
        println("Finished computing the closure")
        return images
    elseif method == :SDE
        if !is_gaussian
            println(" start with predicting the closure ")
            cond = Float32.(cond) |> dev
            images = Float32.(v_test) |> dev
            num_test_samples = size(cond, 4)

            t_range = LinRange(0, 1, num_steps)
            dt = t_range[2] - t_range[1]
            for i in 1:num_steps-1
                t = t_range[i]
                t_sample = Float32.(fill(t, (1,1,1, num_test_samples))) |> dev

                denoiser, st_denoiser = Lux.apply(velocity_cnn, (images, t_sample, cond), ps_denoiser, st_denoiser) |> dev
                score = compute_score_denoiser(t, denoiser) |> dev  # Construct score based on the denoiser
            
                # Compute the drift term
                drift, st_drift = Lux.apply(velocity_cnn, (images, t_sample, cond), ps_drift, st_drift) |> dev
            
                # Construct the forward drift b_F(t, x)
                b_F = drift .+ epsilon(t) .* score |> dev
            
                # Propagate the samples using the Euler-Maruyama method
                images = euler_maruyama(images, b_F, epsilon(t), dt, dev) |> dev
            end
        else
            cond = Float32.(cond) |> dev
            images = randn(size(cond)) |> dev
            num_test_samples = size(cond, 4)

            t_range = LinRange(0, 1, num_steps)
            dt = t_range[2] - t_range[1]
            for i in 1:num_steps-1
                t = t_range[i]
                t_sample = Float32.(fill(t, (1,1,1, num_test_samples))) |> dev

                # Compute the drift term
                drift, st_drift = Lux.apply(velocity_cnn, (images, t_sample, cond), ps_drift, st_drift) |> dev
                # Compute the score
                score = compute_score_velocity(t, drift, images) |> dev
                # compute the term for dt 
                b_F = drift .+ score |> dev
                # Propagate the samples using the Euler-Maruyama method
                images = euler_maruyama(images, b_F, epsilon(t), dt, dev) |> dev
            end
        end
    # Return the generated samples
    println("Finished computing the closure")
    return images
    else
        error("Unknown method: $method. Use :ODE or :SDE.")
    end
end

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
function euler_maruyama(x, b_F, epsilon_t, Δt, dev)
    noise_term = sqrt(2 * epsilon_t) * randn(size(x)) * sqrt(Δt) |> dev
    return x + b_F * Δt + noise_term |> dev
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

function compute_energy_spectra(sol)
    num_trajectories = size(sol, 4);
    
    nx = size(sol, 1);
    ny = size(sol, 2);
    
    kx = fftfreq(nx, nx)
    ky = fftfreq(ny, ny)
    
    K = (kx.^2)' .+ (ky.^2);
    
    K_bins = logrange(1, maximum(K), 100) 
    
    a = 1.6;
    
    energy = zeros(Float32, length(K_bins), num_trajectories) 
    for i = 1:num_trajectories
        u = sol[:, :, 1, i];
        v = sol[:, :, 2, i];

        u_fft = fft(u, [1, 2]);
        v_fft = fft(v, [1, 2]);

        u_fft_squared = abs2.(u_fft) ./ (2 * prod(size(u_fft))^2);
        v_fft_squared = abs2.(v_fft) ./ (2 * prod(size(v_fft))^2);
        
        for j = 1:length(K_bins)
            
            bin = K_bins[j]

            mask = (K .> bin / a) .& (K .< bin * a)
    
            u_fft_filtered = u_fft_squared .* mask
            v_fft_filtered = v_fft_squared .* mask
        
            e = 0.5 * (sum(u_fft_filtered + v_fft_filtered))
    
            energy[j, i] = e
        end
    end
    
    return energy, K_bins
end

function compute_total_energy(sol)

    num_trajectories = size(sol, 5)
    num_time_steps = size(sol, 4)

    energy = zeros(num_time_steps, num_trajectories)
    for i = 1:num_trajectories
        for j in 1:num_time_steps
            energy[j, i] = sum(sol[:, :, 1, j, i].^2) + sum(sol[:, :, 2, j, i].^2)
            energy[j, i] /= 2
        end
    end

    return energy
end

function rk4_with_closure(ubar_test, state_means, state_std, velocity_cnn, ps_drift, st_drift, ps_denoiser, st_denoiser, num_steps, is_gaussian, dev, closure_means, closure_std, dt; time_method=:rk4, method=:ODE)
    stand_ubar_test = standardize_training_set_per_channel(ubar_test, state_means, state_std);
    stand_closure_1 = generate_closure(velocity_cnn, ps_drift, st_drift, ps_denoiser, st_denoiser, stand_ubar_test, stand_ubar_test, num_steps, is_gaussian, dev; time_method=:rk4, method=:ODE);
    closure_1 = inverse_standardize_set_per_channel(stand_closure_1, closure_means, closure_std);
    k1 =  f_les(ubar_test, nothing, 0.0) .+ closure_1;

    input_k2 = ubar_test .+ k1 ./2; 
    stand_input_k2 = standardize_training_set_per_channel(input_k2, state_means, state_std);
    stand_closure_2 = generate_closure(velocity_cnn, ps_drift, st_drift, ps_denoiser, st_denoiser, stand_input_k2, stand_input_k2, num_steps, is_gaussian, dev; time_method=:rk4, method=:ODE);
    closure_2 = inverse_standardize_set_per_channel(stand_closure_2, closure_means, closure_std);
    k2 = f_les(input_k2, nothing, 0.0) .+ closure_2;
    
    input_k3 = ubar_test .+ k2 ./2;
    stand_input_k3 = standardize_training_set_per_channel(input_k3, state_means, state_std);
    stand_closure_3 = generate_closure(velocity_cnn, ps_drift, st_drift, ps_denoiser, st_denoiser, stand_input_k3, stand_input_k3, num_steps, is_gaussian, dev; time_method=:rk4, method=:ODE);
    closure_3 = inverse_standardize_set_per_channel(stand_closure_3, closure_means, closure_std);
    k3 = f_les(input_k3, nothing, 0.0) .+ closure_3;
    
    input_k4 = ubar_test .+ k3;
    stand_input_k4 = standardize_training_set_per_channel(input_k4, state_means, state_std);
    stand_closure_4 = generate_closure(velocity_cnn, ps_drift, st_drift, ps_denoiser, st_denoiser, stand_input_k4, stand_input_k4, num_steps, is_gaussian, dev; time_method=:rk4, method=:ODE);   
    closure_4 = inverse_standardize_set_per_channel(stand_closure_4, closure_means, closure_std);
    k4 = f_les(input_k4, nothing, 0.0) .+ closure_4; 
    
    return ubar_test .+ (dt ./ 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

function rk4_with_closure_deterministic(ubar_test, label, state_means, state_std, velocity_cnn, ps_drift, st_drift, closure_means, closure_std, dt, batch_size, f_les; dev)
    t_sample = Float32.(fill(0, 1, 1, 1, batch_size)) |> dev;
    stand_ubar_test = standardize_training_set_per_channel(ubar_test, state_means, state_std; one_trajectory=true) |> dev;
    stand_closure_1, _ = Lux.apply(velocity_cnn, (stand_ubar_test, t_sample, label), ps_drift, st_drift) |> dev;
    closure_1 = inverse_standardize_set_per_channel(stand_closure_1, closure_means, closure_std; one_trajectory=true) |> dev;
    k1 =  f_les(ubar_test, nothing, 0.0) .+ closure_1;
    input_k2 = ubar_test .+ k1 ./2; 
    stand_input_k2 = standardize_training_set_per_channel(input_k2, state_means, state_std; one_trajectory=true) |> dev;
    stand_closure_2, _ = Lux.apply(velocity_cnn, (stand_input_k2, t_sample, label), ps_drift, st_drift) |> dev;
    closure_2 = inverse_standardize_set_per_channel(stand_closure_2, closure_means, closure_std; one_trajectory=true) |> dev;
    k2 = f_les(input_k2, nothing, 0.0) .+ closure_2;
    input_k3 = ubar_test .+ k2 ./2;
    stand_input_k3 = standardize_training_set_per_channel(input_k3, state_means, state_std; one_trajectory=true) |> dev;
    stand_closure_3, _ = Lux.apply(velocity_cnn, (stand_input_k3, t_sample, label), ps_drift, st_drift) |> dev;
    closure_3 = inverse_standardize_set_per_channel(stand_closure_3, closure_means, closure_std; one_trajectory=true) |> dev;
    k3 = f_les(input_k3, nothing, 0.0) .+ closure_3;
    input_k4 = ubar_test .+ k3;
    stand_input_k4 = standardize_training_set_per_channel(input_k4, state_means, state_std; one_trajectory=true) |> dev;
    stand_closure_4, _ = Lux.apply(velocity_cnn, (stand_input_k4, t_sample, label), ps_drift, st_drift) |> dev;   
    closure_4 = inverse_standardize_set_per_channel(stand_closure_4, closure_means, closure_std; one_trajectory=true) |> dev;
    k4 = f_les(input_k4, nothing, 0.0) .+ closure_4; 
    return ubar_test .+ (dt ./ 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

# Function for inference - add closure over the time
function inference(dt, nt, num_steps, batch_size, v_test, velocity_cnn, ps_drift, _st_drift, ps_denoiser, _st_denoiser, is_gaussian, closure_means, closure_std, state_means, state_std, N_les, Re, dev; time_method=:euler, method=:ODE)
    create_right_hand_side(setup, psolver) = function right_hand_side(u, p, t)
        u = pad_circular(u, 1; dims = 1:2)
        F = INS.momentum(u, nothing, t, setup)
        F = F[2:end-1, 2:end-1, :]
        F = pad_circular(F, 1; dims = 1:2)
        PF = INS.project(F, setup; psolver)
        PF[2:end-1, 2:end-1, :]
    end

    backend = CUDABackend();
    x_les = LinRange(0.0, 1.0, N_les + 1), LinRange(0.0, 1.0, N_les + 1);
    setup_les = INS.Setup(; x=x_les, Re=Re, backend);
    psolver_les = INS.psolver_spectral(setup_les);
    f_les = create_right_hand_side(setup_les, psolver_les); 
    
    global t = 0.0f0; # Initial time t_0
    global u_les = v_test[:,:,:,1]; 
    u_les = reshape(u_les, N_les, N_les, 2, batch_size) |> dev; 
    global ubar_test = u_les; 

    v_test = v_test |> dev; 

    for i=1:nt+1
        global u_les, ubar_test
        if i > 1
            global t
            u_les = step_rk4(u_les, dt, f_les) |> dev;
            next_ubar_test = rk4_with_closure(ubar_test, state_means, state_std, velocity_cnn, ps_drift, _st_drift, ps_denoiser, _st_denoiser, num_steps, is_gaussian, dev, closure_means, closure_std, dt; time_method=:rk4, method=:ODE)
            t += dt
            println("Finished time step ", i)
        end
        
        if i % 5 == 0 && i > 1
            t = (i-1) * dt
            ω_model = Array(INS.vorticity(pad_circular(next_ubar_test, 1; dims = 1:2), setup_les))[:,:,1]
            ω_nomodel = Array(INS.vorticity(pad_circular(u_les, 1; dims=1:2), setup_les))[:,:,1]
            ω_groundtruth =  Array(INS.vorticity(pad_circular(v_test[:,:,:,i], 1; dims=1:2), setup_les))[:,:,1]

            ω_model = ω_model[2:end-1, 2:end-1];
            ω_nomodel = ω_nomodel[2:end-1, 2:end-1];
            ω_groundtruth = ω_groundtruth[2:end-1, 2:end-1];
             
            ω_closure = abs.(ω_groundtruth - ω_nomodel);
            ω_pred_closure = abs.(ω_model - ω_nomodel);
            ω_error_map = abs.(ω_model - ω_groundtruth);

            title_model = @sprintf("Vorticity model, t = %.3f", t)
            title_nomodel = @sprintf("Vorticity no model, t=%.3f", t)
            title_groundtruth = @sprintf("Vorticity ground truth, t=%.3f", t)

            title_closure = @sprintf("vorticity closure, t=%.3f", t)
            title_pred_closure = @sprintf("Predicted closure, t=%.3f", t)
            title_error = @sprintf("Error model, t=%.3f", t)

            # Determine the global color scale range
            all_data_state = [ω_model, ω_nomodel, ω_groundtruth]
            all_data_closure = [ω_closure, ω_pred_closure, ω_error_map]
            v_min_state = minimum([minimum(data) for data in all_data_state])
            v_max_state = maximum([maximum(data) for data in all_data_state])
            v_min_closure = minimum([minimum(data) for data in all_data_closure])
            v_max_closure = maximum([maximum(data) for data in all_data_closure])

            p1 = Plots.heatmap(ω_nomodel'; xlabel = "x", ylabel="y", title=title_nomodel, color=:viridis, clim = (v_min_state, v_max_state))
            p2 = Plots.heatmap(ω_model'; xlabel = "x", ylabel = "y", title=title_model, color=:viridis, clim = (v_min_state, v_max_state))
            p3 = Plots.heatmap(ω_groundtruth'; xlabel = "x", ylabel = "y", title=title_groundtruth, color=:viridis, clim = (v_min_state, v_max_state))
            p4 = Plots.heatmap(ω_closure'; xlabel="x", ylabel="y", title=title_closure, color=:viridis, clim = (v_min_closure, v_max_closure))
            p5 = Plots.heatmap(ω_pred_closure'; xlabel="x", ylabel="y", title=title_pred_closure, color=:viridis, clim = (v_min_closure, v_max_closure))
            p6 = Plots.heatmap(ω_error_map'; xlabel="x", ylabel="y", title=title_error, color=:viridis, clim = (v_min_closure, v_max_closure))

            fig = Plots.plot(p1, p2, p3, p4, p5, p6, layout = (2, 3), size=(3200, 1200))
            savefig(fig, @sprintf("figures/vorticity_timestep_%03d.png", i))

            # Print the computed error
            println("Error between model and ground truth at t = ", t, ": ", mean(ω_error_map))
        end
    end
end

function inference_deterministic(dt, nt, batch_size, v_test, velocity_cnn, ps_drift, _st_drift, closure_means, closure_std, state_means, state_std, N_les, Re; dev)
    create_right_hand_side(setup, psolver) = function right_hand_side(u, p, t)
        u = pad_circular(u, 1; dims = 1:2)
        F = INS.momentum(u, nothing, t, setup)
        F = F[2:end-1, 2:end-1, :]
        F = pad_circular(F, 1; dims = 1:2)
        PF = INS.project(F, setup; psolver)
        PF[2:end-1, 2:end-1, :]
    end

    backend = CUDABackend();
    x_les = LinRange(0.0, 1.0, N_les + 1), LinRange(0.0, 1.0, N_les + 1);
    setup_les = INS.Setup(; x=x_les, Re=Re, backend);
    psolver_les = INS.psolver_spectral(setup_les);
    f_les = create_right_hand_side(setup_les, psolver_les); 
    
    global t = 0.0f0; 
    global u_les = v_test[:,:,:,1]; 
    u_les = reshape(u_les, N_les, N_les, 2, batch_size) |> dev; 
    global ubar_test = u_les |> dev; 
    t_sample = Float32.(fill(0, 1, 1, 1, batch_size)) |> dev

    for i=1:nt+1
        global u_les, ubar_test
        if i > 1
            global t
            u_les = step_rk4(u_les, dt, f_les) |> dev;
            stand_input = standardize_training_set_per_channel(ubar_test, state_means, state_std; one_trajectory=true) |> dev;
            stand_closure, _ = Lux.apply(velocity_cnn, (stand_input, t_sample), ps_drift, _st_drift) |> dev;   
            closure = inverse_standardize_set_per_channel(stand_closure, closure_means, closure_std; one_trajectory=true) |> dev;
            ubar_test = step_rk4(ubar_test, dt, f_les) .+ (dt .* closure)
            # ubar_test = rk4_with_closure_deterministic(ubar_test, ubar_test, state_means, state_std, velocity_cnn, ps_drift, _st_drift, closure_means, closure_std, dt, batch_size, f_les; dev) |> dev;
            t += dt
        end
        
        if i % 50 == 0 # && i > 1
            t = (i-1) * dt
            groundtruth = v_test[:,:,:,i] |> dev;
            ω_model = Array(INS.vorticity(pad_circular(ubar_test, 1; dims = 1:2), setup_les))[:,:,1]
            ω_nomodel = Array(INS.vorticity(pad_circular(u_les, 1; dims=1:2), setup_les))[:,:,1]
            ω_groundtruth =  Array(INS.vorticity(pad_circular(groundtruth, 1; dims=1:2), setup_les))[:,:,1]

            ω_model = ω_model[2:end-1, 2:end-1];
            ω_nomodel = ω_nomodel[2:end-1, 2:end-1];
            ω_groundtruth = ω_groundtruth[2:end-1, 2:end-1];
             
            ω_closure = abs.(ω_groundtruth - ω_nomodel);
            ω_pred_closure = abs.(ω_model - ω_nomodel);
            ω_error_map = abs.(ω_model - ω_groundtruth);

            # Compute energy spectra
            energy_nomodel, K_bins = compute_energy_spectra(Array(u_les))
            energy_model, _ = compute_energy_spectra(Array(ubar_test))
            energy_groundtruth, _ = compute_energy_spectra(Array(groundtruth))

            title_model = @sprintf("Vorticity model, t = %.3f", t)
            title_nomodel = @sprintf("Vorticity no model, t=%.3f", t)
            title_groundtruth = @sprintf("Vorticity ground truth, t=%.3f", t)

            title_closure = @sprintf("vorticity closure, t=%.3f", t)
            title_pred_closure = @sprintf("Predicted closure, t=%.3f", t)
            title_error = @sprintf("Error model, t=%.3f", t)

            # Determine the global color scale range
            all_data_state = [ω_model, ω_nomodel, ω_groundtruth]
            all_data_closure = [ω_closure, ω_pred_closure, ω_error_map]
            v_min_state = minimum([minimum(data) for data in all_data_state])
            v_max_state = maximum([maximum(data) for data in all_data_state])
            v_min_closure = minimum([minimum(data) for data in all_data_closure])
            v_max_closure = maximum([maximum(data) for data in all_data_closure])

            p1 = Plots.heatmap(ω_nomodel'; xlabel = "x", ylabel="y", title=title_nomodel, color=:viridis, clim = (v_min_state, v_max_state))
            p2 = Plots.heatmap(ω_model'; xlabel = "x", ylabel = "y", title=title_model, color=:viridis, clim = (v_min_state, v_max_state))
            p3 = Plots.heatmap(ω_groundtruth'; xlabel = "x", ylabel = "y", title=title_groundtruth, color=:viridis, clim = (v_min_state, v_max_state))
            p4 = Plots.heatmap(ω_closure'; xlabel="x", ylabel="y", title=title_closure, color=:viridis, clim = (v_min_closure, v_max_closure))
            p5 = Plots.heatmap(ω_pred_closure'; xlabel="x", ylabel="y", title=title_pred_closure, color=:viridis, clim = (v_min_closure, v_max_closure))
            p6 = Plots.heatmap(ω_error_map'; xlabel="x", ylabel="y", title=title_error, color=:viridis, clim = (v_min_closure, v_max_closure))

            # Plot energy spectra
            energy_spectrum_plot = Plots.plot(K_bins, energy_nomodel[:,1], label="No Model", xaxis=:log, yaxis=:log,
                                              xlabel="Wavenumber k", ylabel="Energy E(k)",
                                              title="Energy Spectrum at t=$(round(t, digits=3))")
            Plots.plot!(energy_spectrum_plot, K_bins, energy_model[:,1], label="Model")
            Plots.plot!(energy_spectrum_plot, K_bins, energy_groundtruth[:,1], label="Ground Truth")

            # Combine all plots into one figure
            combined_fig = Plots.plot(p1, p2, p3, p4, p5, p6, energy_spectrum_plot, layout=(3, 3), size=(3200, 2400))
            savefig(combined_fig, @sprintf("figures/deterministic_model/combined_timestep_%03d.png", i))

            # Print the computed error
            println("Error between model and ground truth at t = ", t, ": ", mean(ω_error_map))
        end
    end
end

function inference_deterministic_without_ground_truth(dt, nt, batch_size, v_test, velocity_cnn, ps_drift, _st_drift, closure_means, closure_std, state_means, state_std, N_les, Re; dev)
    create_right_hand_side(setup, psolver) = function right_hand_side(u, p, t)
        u = pad_circular(u, 1; dims = 1:2)
        F = INS.momentum(u, nothing, t, setup)
        F = F[2:end-1, 2:end-1, :]
        F = pad_circular(F, 1; dims = 1:2)
        PF = INS.project(F, setup; psolver)
        PF[2:end-1, 2:end-1, :]
    end

    backend = CUDABackend();
    x_les = LinRange(0.0, 1.0, N_les + 1), LinRange(0.0, 1.0, N_les + 1);
    setup_les = INS.Setup(; x=x_les, Re=Re, backend);
    psolver_les = INS.psolver_spectral(setup_les);
    f_les = create_right_hand_side(setup_les, psolver_les); 
    
    global t = 0.0f0; 
    global u_les = v_test[:,:,:,1]; 
    u_les = reshape(u_les, N_les, N_les, 2, batch_size) |> dev; 
    global ubar_test = u_les |> dev; 
    t_sample = Float32.(fill(0, 1, 1, 1, batch_size)) |> dev

    for i=1:nt+1
        global u_les, ubar_test
        if i > 1
            global t
            u_les = step_rk4(u_les, dt, f_les) |> dev;
            stand_input = standardize_training_set_per_channel(ubar_test, state_means, state_std; one_trajectory=true) |> dev;
            stand_closure, _ = Lux.apply(velocity_cnn, (stand_input, t_sample), ps_drift, _st_drift) |> dev;   
            closure = inverse_standardize_set_per_channel(stand_closure, closure_means, closure_std; one_trajectory=true) |> dev;
            ubar_test = step_rk4(ubar_test, dt, f_les) .+ (dt .* closure)
            # ubar_test = rk4_with_closure_deterministic(ubar_test, ubar_test, state_means, state_std, velocity_cnn, ps_drift, _st_drift, closure_means, closure_std, dt, batch_size, f_les; dev) |> dev;
            t += dt
        end
        
        if i % 100 == 0 # && i > 1
            t = (i-1) * dt
            ω_model = Array(INS.vorticity(pad_circular(ubar_test, 1; dims = 1:2), setup_les))[:,:,1]
            ω_nomodel = Array(INS.vorticity(pad_circular(u_les, 1; dims=1:2), setup_les))[:,:,1]
            ω_model = ω_model[2:end-1, 2:end-1];
            ω_nomodel = ω_nomodel[2:end-1, 2:end-1];
            ω_pred_closure = abs.(ω_model - ω_nomodel);
            energy_nomodel, K_bins = compute_energy_spectra(Array(u_les))
            energy_model, _ = compute_energy_spectra(Array(ubar_test))
            title_model = @sprintf("Vorticity model, t = %.3f", t)
            title_nomodel = @sprintf("Vorticity no model, t=%.3f", t)
            title_pred_closure = @sprintf("Predicted closure, t=%.3f", t)

            all_data_state = [ω_model, ω_nomodel]
            v_min_state = minimum([minimum(data) for data in all_data_state])
            v_max_state = maximum([maximum(data) for data in all_data_state])

            p1 = Plots.heatmap(ω_nomodel'; xlabel = "x", ylabel="y", title=title_nomodel, color=:viridis, clim = (v_min_state, v_max_state))
            p2 = Plots.heatmap(ω_model'; xlabel = "x", ylabel = "y", title=title_model, color=:viridis, clim = (v_min_state, v_max_state))
            p3 = Plots.heatmap(ω_pred_closure'; xlabel="x", ylabel="y", title=title_pred_closure, color=:viridis) 

            # Plot energy spectra
            energy_spectrum_plot = Plots.plot(K_bins, energy_nomodel[:,1], label="No Model", xaxis=:log, yaxis=:log,
                                              xlabel="Wavenumber k", ylabel="Energy E(k)",
                                              title="Energy Spectrum at t=$(round(t, digits=3))")
            Plots.plot!(energy_spectrum_plot, K_bins, energy_model[:,1], label="Model")

            # Combine all plots into one figure
            combined_fig = Plots.plot(p1, p2, p3, energy_spectrum_plot, layout=(2, 2), size=(2400, 2400))
            savefig(combined_fig, @sprintf("figures/deterministic_model/inf_combined_timestep_%03d.png", i))
        end
    end
end