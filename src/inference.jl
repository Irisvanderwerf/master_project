using Plots
using IncompressibleNavierStokes
const INS = IncompressibleNavierStokes
using OrdinaryDiffEq
using FFTW
using KernelAbstractions
using Printf

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

function inference_tunable_diffusion(dt, nt, batch_size, v_test, c_test, velocity_cnn, ps_drift, _st_drift, closure_means, closure_std, state_means, state_std, N_les, Re, num_initial_conditions, num_train_conditions, dev)
    num_steps = 200;

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

    trajectory_to_evaluate = rand(1:(num_initial_conditions-num_train_conditions)) |> dev;
    v_test = v_test[:,:,:,:,trajectory_to_evaluate] |> dev;
    c_test = c_test[:,:,:,:,trajectory_to_evaluate] |> dev;
    
    global t = 0.0f0; 
    global u_les = v_test[:,:,:,1]; 
    global closure = c_test[:,:,:,1];
    u_les = reshape(u_les, N_les, N_les, 2, batch_size) |> dev; 
    u_model = u_les |> dev;
    closure = reshape(closure, N_les, N_les, 2, batch_size) |> dev;

    for i=1:nt+1
        global u_les
        if i > 1
            global t
            u_les = step_rk4(u_les, dt, f_les) |> dev;

            stand_state_cond = standardize_training_set_per_channel(u_model, state_means, state_std; one_trajectory=true) |> dev;
            stand_closure = standardize_training_set_per_channel(closure, closure_means, closure_std; one_trajectory=true) |> dev;

            up_stand_closure = generate_closure_with_tunable_diffusion(velocity_cnn, ps_drift, _st_drift, stand_closure, stand_state_cond, stand_closure, num_steps, dev) |> dev; 
            closure = inverse_standardize_set_per_channel(up_stand_closure, closure_means, closure_std; one_trajectory=true) |> dev;
            u_model = step_rk4(u_model, dt, f_les) .+ (dt .* closure) |> dev;
            t += dt
        end

        if i % 200 == 0 && i > 1
            t = (i-1) * dt
            groundtruth = v_test[:,:,:,i] |> dev;
            ω_model = Array(INS.vorticity(pad_circular(u_model, 1; dims = 1:2), setup_les))[:,:,1]
            ω_nomodel = Array(INS.vorticity(pad_circular(u_les, 1; dims=1:2), setup_les))[:,:,1]
            ω_groundtruth =  Array(INS.vorticity(pad_circular(groundtruth, 1; dims=1:2), setup_les))[:,:,1]
            
            ω_model = ω_model[2:end-1, 2:end-1];
            ω_nomodel = ω_nomodel[2:end-1, 2:end-1];
            ω_groundtruth = ω_groundtruth[2:end-1, 2:end-1];
                         
            ω_closure = abs.(ω_groundtruth - ω_nomodel);
            ω_pred_closure = abs.(ω_model - ω_nomodel);
            ω_error_map = abs.(ω_model - ω_groundtruth);
            
            energy_nomodel, K_bins = compute_energy_spectra(Array(u_les))
            energy_model, _ = compute_energy_spectra(Array(u_model))
            energy_groundtruth, _ = compute_energy_spectra(Array(groundtruth))
            
            title_model = @sprintf("Vorticity model, t = %.3f", t)
            title_nomodel = @sprintf("Vorticity no model, t=%.3f", t)
            title_groundtruth = @sprintf("Vorticity ground truth, t=%.3f", t)
            
            title_closure = @sprintf("vorticity closure, t=%.3f", t)
            title_pred_closure = @sprintf("Predicted closure, t=%.3f", t)
            title_error = @sprintf("Error model, t=%.3f", t)
            
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
            
            energy_spectrum_plot = Plots.plot(K_bins, energy_nomodel[:,1], label="No Model", xaxis=:log, yaxis=:log,
                xlabel="Wavenumber k", ylabel="Energy E(k)",
                title="Energy Spectrum at t=$(round(t, digits=3))")
            Plots.plot!(energy_spectrum_plot, K_bins, energy_model[:,1], label="Model")
            Plots.plot!(energy_spectrum_plot, K_bins, energy_groundtruth[:,1], label="Ground Truth")
            
            combined_fig = Plots.plot(p1, p2, p3, p4, p5, p6, energy_spectrum_plot, layout=(3, 3), size=(3200, 2400))
            savefig(combined_fig, @sprintf("figures/SDE_model/combined_timestep_%03d.png", i))
            
            println("Error between model and ground truth at t = ", t, ": ", mean(ω_error_map))
            println("Error between LES and filtered DNS at t = ", t, ": ", mean(ω_closure))

        end
    end
end

function inference_mean_closure(dt, nt, batch_size, v_test, c_test, velocity_cnn, ps_drift, _st_drift, closure_means, closure_std, state_means, state_std, N_les, Re, num_initial_conditions, num_train_conditions, dev)
    num_steps = 200;

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

    trajectory_to_evaluate = rand(1:(num_initial_conditions-num_train_conditions)) |> dev;
    v_test = v_test[:,:,:,:,trajectory_to_evaluate] |> dev;
    c_test = c_test[:,:,:,:,trajectory_to_evaluate] |> dev;
    
    global t = 0.0f0; 
    global u_les = v_test[:,:,:,1]; 
    global closure = c_test[:,:,:,1];
    u_les = reshape(u_les, N_les, N_les, 2, batch_size) |> dev; 
    u_model = u_les |> dev;
    closure = reshape(closure, N_les, N_les, 2, batch_size) |> dev;

    for i=1:nt+1
        global u_les
        if i > 1
            global t
            # different dt for les 
            u_les = step_rk4(u_les, dt, f_les) |> dev;

            stand_state_cond = standardize_training_set_per_channel(u_model, state_means, state_std; one_trajectory=true) |> dev;
            stand_closure = standardize_training_set_per_channel(closure, closure_means, closure_std; one_trajectory=true) |> dev;

            up_stand_closure = generate_closure(velocity_cnn, ps_drift, _st_drift, stand_closure, stand_state_cond, stand_closure, num_steps, dev) |> dev; 
            closure = inverse_standardize_set_per_channel(up_stand_closure, closure_means, closure_std; one_trajectory=true) |> dev;
            # add projection after adding closure
            u_model = step_rk4(u_model, dt, f_les) .+ (dt .* closure) |> dev;
            t += dt
        end

        if i % 200 == 0 && i > 1
            t = (i-1) * dt
            groundtruth = v_test[:,:,:,i] |> dev;
            ω_model = Array(INS.vorticity(pad_circular(u_model, 1; dims = 1:2), setup_les))[:,:,1]
            ω_nomodel = Array(INS.vorticity(pad_circular(u_les, 1; dims=1:2), setup_les))[:,:,1]
            ω_groundtruth =  Array(INS.vorticity(pad_circular(groundtruth, 1; dims=1:2), setup_les))[:,:,1]
            
            ω_model = ω_model[2:end-1, 2:end-1];
            ω_nomodel = ω_nomodel[2:end-1, 2:end-1];
            ω_groundtruth = ω_groundtruth[2:end-1, 2:end-1];
                         
            ω_closure = abs.(ω_groundtruth - ω_nomodel);
            ω_pred_closure = abs.(ω_model - ω_nomodel);
            ω_error_map = abs.(ω_model - ω_groundtruth);
            
            energy_nomodel, K_bins = compute_energy_spectra(Array(u_les))
            energy_model, _ = compute_energy_spectra(Array(u_model))
            energy_groundtruth, _ = compute_energy_spectra(Array(groundtruth))
            
            title_model = @sprintf("Vorticity model, t = %.3f", t)
            title_nomodel = @sprintf("Vorticity no model, t=%.3f", t)
            title_groundtruth = @sprintf("Vorticity ground truth, t=%.3f", t)
            
            title_closure = @sprintf("vorticity closure, t=%.3f", t)
            title_pred_closure = @sprintf("Predicted closure, t=%.3f", t)
            title_error = @sprintf("Error model, t=%.3f", t)
            
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
            
            energy_spectrum_plot = Plots.plot(K_bins, energy_nomodel[:,1], label="No Model", xaxis=:log, yaxis=:log,
                xlabel="Wavenumber k", ylabel="Energy E(k)",
                title="Energy Spectrum at t=$(round(t, digits=3))")
            Plots.plot!(energy_spectrum_plot, K_bins, energy_model[:,1], label="Model")
            Plots.plot!(energy_spectrum_plot, K_bins, energy_groundtruth[:,1], label="Ground Truth")
            
            combined_fig = Plots.plot(p1, p2, p3, p4, p5, p6, energy_spectrum_plot, layout=(3, 3), size=(3200, 2400))
            savefig(combined_fig, @sprintf("figures/SDE_model/combined_timestep_%03d.png", i))
            
            println("Error between model and ground truth at t = ", t, ": ", mean(ω_error_map))
            println("Error between LES and filtered DNS at t = ", t, ": ", mean(ω_closure))
        end
    end
end

function inference_mean(dt, nt, batch_size, v_test, c_test, velocity_cnn, ps_drift, _st_drift, closure_means, closure_std, state_means, state_std, N_les, Re, num_initial_conditions, num_train_conditions, dev)
    num_steps = 200;

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

    trajectory_to_evaluate = rand(1:(num_initial_conditions-num_train_conditions)) |> dev;
    v_test = v_test[:,:,:,:,trajectory_to_evaluate] |> dev;
    c_test = c_test[:,:,:,:,trajectory_to_evaluate] |> dev;
    
    global t = 0.0f0; 
    global u_les = v_test[:,:,:,1]; 
    global closure = c_test[:,:,:,1];
    u_les = reshape(u_les, N_les, N_les, 2, 1) |> dev; 
    mean_u_model = u_les |> dev; 
    mean_closure = reshape(closure, N_les, N_les, 2, 1) |> dev; 
 
    for i=1:nt+1
        global u_les
        if i > 1
            global t
            u_les = step_rk4(u_les, dt, f_les) |> dev; 

            u_model = repeat(mean_u_model, 1, 1, 1, batch_size) |> dev;
            closure = repeat(mean_closure, 1, 1, 1, batch_size) |> dev;

            stand_state_cond = standardize_training_set_per_channel(u_model, state_means, state_std; one_trajectory=true) |> dev;
            stand_closure = standardize_training_set_per_channel(closure, closure_means, closure_std; one_trajectory=true) |> dev; 

            up_stand_closure = generate_closure(velocity_cnn, ps_drift, _st_drift, stand_closure, stand_state_cond, stand_closure, num_steps, dev) |> dev; 
            closure = inverse_standardize_set_per_channel(up_stand_closure, closure_means, closure_std; one_trajectory=true) |> dev; 
            
            mean_closure = mean(closure, dims=4) |> dev;
            mean_u_model = mean(u_model, dims=4) |> dev;
            
            mean_u_model = step_rk4(mean_u_model, dt, f_les) .+ (dt .* mean_closure) |> dev;
            t += dt
        end

        if i % 20 == 0 && i > 1
            t = (i-1) * dt
            groundtruth = v_test[:,:,:,i] |> dev;
            ω_model = Array(INS.vorticity(pad_circular(mean_u_model, 1; dims = 1:2), setup_les))[:,:,1]
            ω_nomodel = Array(INS.vorticity(pad_circular(u_les, 1; dims=1:2), setup_les))[:,:,1]
            ω_groundtruth =  Array(INS.vorticity(pad_circular(groundtruth, 1; dims=1:2), setup_les))[:,:,1]
            
            ω_model = ω_model[2:end-1, 2:end-1];
            ω_nomodel = ω_nomodel[2:end-1, 2:end-1];
            ω_groundtruth = ω_groundtruth[2:end-1, 2:end-1];
                         
            ω_closure = abs.(ω_groundtruth - ω_nomodel);
            ω_pred_closure = abs.(ω_model - ω_nomodel);
            ω_error_map = abs.(ω_model - ω_groundtruth);
            
            energy_nomodel, K_bins = compute_energy_spectra(Array(u_les))
            energy_model, _ = compute_energy_spectra(Array(mean_u_model))
            energy_groundtruth, _ = compute_energy_spectra(Array(groundtruth))
            
            title_model = @sprintf("Vorticity model, t = %.3f", t)
            title_nomodel = @sprintf("Vorticity no model, t=%.3f", t)
            title_groundtruth = @sprintf("Vorticity ground truth, t=%.3f", t)
            
            title_closure = @sprintf("vorticity closure, t=%.3f", t)
            title_pred_closure = @sprintf("Predicted closure, t=%.3f", t)
            title_error = @sprintf("Error model, t=%.3f", t)
            
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
            
            energy_spectrum_plot = Plots.plot(K_bins, energy_nomodel[:,1], label="No Model", xaxis=:log, yaxis=:log,
                xlabel="Wavenumber k", ylabel="Energy E(k)",
                title="Energy Spectrum at t=$(round(t, digits=3))")
            Plots.plot!(energy_spectrum_plot, K_bins, energy_model[:,1], label="Model")
            Plots.plot!(energy_spectrum_plot, K_bins, energy_groundtruth[:,1], label="Ground Truth")
            
            combined_fig = Plots.plot(p1, p2, p3, p4, p5, p6, energy_spectrum_plot, layout=(3, 3), size=(3200, 2400))
            savefig(combined_fig, @sprintf("figures/SDE_model/combined_timestep_%03d.png", i))
            
            println("Error between model and ground truth at t = ", t, ": ", mean(ω_error_map))
            println("Error between LES and filtered DNS at t = ", t, ": ", mean(ω_closure))
        end
    end
end

function inference(dt, nt, batch_size, v_test, c_test, velocity_cnn, ps_drift, _st_drift, closure_means, closure_std, state_means, state_std, N_les, Re, num_initial_conditions, num_train_conditions, dev)
    num_steps = 200;

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

    trajectory_to_evaluate = rand(1:(num_initial_conditions-num_train_conditions)) |> dev;
    v_test = v_test[:,:,:,:,trajectory_to_evaluate] |> dev;
    c_test = c_test[:,:,:,:,trajectory_to_evaluate] |> dev;
    
    global t = 0.0f0; 
    global u_les = v_test[:,:,:,1]; 
    global closure = c_test[:,:,:,1];
    u_les = reshape(u_les, N_les, N_les, 2, batch_size) |> dev; 
    u_model = u_les |> dev;
    closure = reshape(closure, N_les, N_les, 2, batch_size) |> dev;

    for i=1:nt+1
        global u_les
        if i > 1
            global t
            u_les = step_rk4(u_les, dt, f_les) |> dev;

            stand_state_cond = standardize_training_set_per_channel(u_model, state_means, state_std; one_trajectory=true) |> dev;
            stand_closure = standardize_training_set_per_channel(closure, closure_means, closure_std; one_trajectory=true) |> dev;

            up_stand_closure = generate_closure(velocity_cnn, ps_drift, _st_drift, stand_closure, stand_state_cond, stand_closure, num_steps, dev) |> dev; 
            closure = inverse_standardize_set_per_channel(up_stand_closure, closure_means, closure_std; one_trajectory=true) |> dev;
            u_model = step_rk4(u_model, dt, f_les) .+ (dt .* closure) |> dev;
            t += dt
        end

        if i % 200 == 0 && i > 1
            t = (i-1) * dt
            groundtruth = v_test[:,:,:,i] |> dev;
            ω_model = Array(INS.vorticity(pad_circular(u_model, 1; dims = 1:2), setup_les))[:,:,1]
            ω_nomodel = Array(INS.vorticity(pad_circular(u_les, 1; dims=1:2), setup_les))[:,:,1]
            ω_groundtruth =  Array(INS.vorticity(pad_circular(groundtruth, 1; dims=1:2), setup_les))[:,:,1]
            
            ω_model = ω_model[2:end-1, 2:end-1];
            ω_nomodel = ω_nomodel[2:end-1, 2:end-1];
            ω_groundtruth = ω_groundtruth[2:end-1, 2:end-1];
                         
            ω_closure = abs.(ω_groundtruth - ω_nomodel);
            ω_pred_closure = abs.(ω_model - ω_nomodel);
            ω_error_map = abs.(ω_model - ω_groundtruth);
            
            energy_nomodel, K_bins = compute_energy_spectra(Array(u_les))
            energy_model, _ = compute_energy_spectra(Array(u_model))
            energy_groundtruth, _ = compute_energy_spectra(Array(groundtruth))
            
            title_model = @sprintf("Vorticity model, t = %.3f", t)
            title_nomodel = @sprintf("Vorticity no model, t=%.3f", t)
            title_groundtruth = @sprintf("Vorticity ground truth, t=%.3f", t)
            
            title_closure = @sprintf("vorticity closure, t=%.3f", t)
            title_pred_closure = @sprintf("Predicted closure, t=%.3f", t)
            title_error = @sprintf("Error model, t=%.3f", t)
            
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
            
            energy_spectrum_plot = Plots.plot(K_bins, energy_nomodel[:,1], label="No Model", xaxis=:log, yaxis=:log,
                xlabel="Wavenumber k", ylabel="Energy E(k)",
                title="Energy Spectrum at t=$(round(t, digits=3))")
            Plots.plot!(energy_spectrum_plot, K_bins, energy_model[:,1], label="Model")
            Plots.plot!(energy_spectrum_plot, K_bins, energy_groundtruth[:,1], label="Ground Truth")
            
            combined_fig = Plots.plot(p1, p2, p3, p4, p5, p6, energy_spectrum_plot, layout=(3, 3), size=(3200, 2400))
            savefig(combined_fig, @sprintf("figures/SDE_model/combined_timestep_%03d.png", i))
            
            println("Error between model and ground truth at t = ", t, ": ", mean(ω_error_map))
            println("Error between LES and filtered DNS at t = ", t, ": ", mean(ω_closure))
        end
    end
end