using Plots
using IncompressibleNavierStokes
const INS = IncompressibleNavierStokes
using OrdinaryDiffEq
using FFTW
using KernelAbstractions

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

function runge_kutta_4(velocity_cnn, ps, st, images, cond,  t, dt, num_test_samples, dev)
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

# Function plot the energy spectrum
function compute_energy_spectrum(u_field)
    u_x, u_y = u_field[:,:,1], u_field[:,:,2]
    N_x, N_y = size(u_x)
    kx = fftshift(-N_x ÷ 2:(N_x ÷ 2 - 1)) * (2π / N_x)
    ky = fftshift(-N_y ÷ 2:(N_y ÷ 2 - 1)) * (2π / N_y)
    KX = reshape(kx, 1, :)  # Row vector
    KY = reshape(ky, :, 1)  # Column vector
    k = sqrt.(KX.^2 .+ KY.^2)
    u_x_hat = fftshift(fft(u_x))
    u_y_hat = fftshift(fft(u_y))
    E_k = 0.5 * (abs.(u_x_hat).^2 .+ abs.(u_y_hat).^2)
    return k, E_k
end

# Bin the energy spectrum radially
function radial_binning(k, E_k, N_bins=50)
    k_max = maximum(k)
    k_bins = range(0, stop=k_max, length=N_bins+1)
    E_spectrum = zeros(N_bins)
    for j in 1:N_bins
        in_bin = (k .>= k_bins[j]) .& (k .< k_bins[j+1])
        E_spectrum[j] = sum(E_k[in_bin])
    end
        k_bin_centers = 0.5 .* (k_bins[1:end-1] + k_bins[2:end])
        return k_bin_centers, E_spectrum
end

# Function for inference - add closure over the time
function inference(dt, nt, num_steps, batch_size, v_test_standardized, v_test, velocity_cnn, ps_drift, _st_drift, ps_denoiser, _st_denoiser, is_gaussian, closure_means, closure_std, state_means, state_std, N_les, Re, dev; time_method=:euler, method=:ODE)
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
    global ubar_test = v_test_standardized[:,:,:,1];
    ubar_test = reshape(ubar_test, N_les, N_les, 2, batch_size) |> dev;  
    global u_les = v_test[:,:,:,1]; 
    u_les = reshape(ubar_test, N_les, N_les, 2, batch_size) |> dev;  

    for i=1:nt+1
        global u_les, ubar_test
        if i > 1
            global t
            stand_closure = generate_closure(velocity_cnn, ps_drift, _st_drift, ps_denoiser, _st_denoiser, ubar_test, ubar_test, num_steps, is_gaussian, dev; time_method=:euler, method=:ODE) |> dev; 
            closure = inverse_standardize_set_per_channel(stand_closure, closure_means, closure_std) |> dev
            u_les = u_les .+ (dt .* f_les(u_les, nothing, 0.0)) # Forward Euler step
            next_ubar_test = u_les .+ (dt .* closure)
            ubar_test = standardize_training_set_per_channel(next_ubar_test, state_means, state_std)
            t += dt
            println("Finished time step ", i)
        end
        
        if i % 10 == 0 && i > 1
            t = (i-1) * dt
            ω_model = Array(INS.vorticity(pad_circular(next_ubar_test, 1; dims = 1:2), setup_les))[:,:,1]
            ω_nomodel = Array(INS.vorticity(pad_circular(u_les, 1; dims=1:2), setup_les))[:,:,1]
            ω_groundtruth =  Array(INS.vorticity(pad_circular(v_test[:,:,:,i], 1; dims=1:2), setup_les))[:,:,1]

            ω_model = ω_model[2:end-1, 2:end-1];
            ω_nomodel = ω_nomodel[2:end-1, 2:end-1];
            ω_groundtruth = ω_groundtruth[2:end-1, 2:end-1];

            k_les, E_les = compute_energy_spectrum(u_les[:,:,:,1])
            k_next, E_next = compute_energy_spectrum(next_ubar_test[:,:,:,1])
            k_truth, E_truth = compute_energy_spectrum(v_test[:,:,:,i])
            k_les_bins, E_les_spectrum = radial_binning(k_les, E_les)
            k_next_bins, E_next_spectrum = radial_binning(k_next, E_next)
            k_truth_bins, E_truth_spectrum = radial_binning(k_truth, E_truth)

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
            all_data = [ω_model, ω_nomodel, ω_groundtruth, ω_closure, ω_pred_closure, ω_error_map]
            v_min = minimum([minimum(data) for data in all_data])
            v_max = maximum([maximum(data) for data in all_data])

            p1 = Plots.heatmap(ω_nomodel'; xlabel = "x", ylabel="y", title=title_nomodel, color=:viridis, clim = (v_min, v_max))
            p2 = Plots.heatmap(ω_model'; xlabel = "x", ylabel = "y", title=title_model, color=:viridis, clim = (v_min, v_max))
            p3 = Plots.heatmap(ω_groundtruth'; xlabel = "x", ylabel = "y", title=title_groundtruth, color=:viridis, clim = (v_min, v_max))
            p4 = Plots.heatmap(ω_closure'; xlabel="x", ylabel="y", title=title_closure, color=:viridis, clim = (v_min, v_max))
            p5 = Plots.heatmap(ω_pred_closure'; xlabel="x", ylabel="y", title=title_pred_closure, color=:viridis, clim = (v_min, v_max))
            p6 = Plots.heatmap(ω_error_map'; xlabel="x", ylabel="y", title=title_error, color=:viridis, clim = (v_min, v_max))
            p7 = Plots.plot(k_les_bins, E_les_spectrum, xlabel="k", ylabel="E(k)", label="no model", xscale=:log10, yscale=:log10, title="Energy Spectrum")
            Plots.plot!(p7, k_next_bins, E_next_spectrum, label="model")
            Plots.plot!(p7, k_truth_bins, E_truth_spectrum, label="ground truth")

        
            fig = Plots.plot(p1, p2, p3, p4, p5, p6, p7, layout = (3, 3), size=(3200, 1200))
            savefig(fig, @sprintf("figures/vorticity_timestep_%03d.png", i))

            # Print the computed error
            println("Error between model and ground truth at t = ", t, ": ", mean(ω_error_map))
        end
    end
end

