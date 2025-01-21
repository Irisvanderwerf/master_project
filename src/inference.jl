using Plots
using IncompressibleNavierStokes
const INS = IncompressibleNavierStokes
using OrdinaryDiffEq
using FFTW
using KernelAbstractions
using Printf

function compute_energy_spectra(sol)
    nx = size(sol, 1);
    ny = size(sol, 2);

    kx = fftfreq(nx, nx)
    ky = fftfreq(ny, ny)
    
    K = (kx.^2)' .+ (ky.^2);
    
    K_bins = logrange(1, maximum(K), 100) 
    
    a = 1.6;
    
    energy = zeros(Float32, length(K_bins)) 
    u = sol[:, :, 1];
    v = sol[:, :, 2];

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
    
        energy[j] = e
    end
    
    return energy, K_bins
end

function compute_total_energy(sol::CuArray)
    if ndims(sol) != 4
        throw(ArgumentError("Input `sol` must have 4 dimensions (nx, ny, components, time_steps). Got: $(size(sol))"))
    end
    energy_per_point = sum(abs2.(sol[:, :, :, :]), dims=3)
    total_energy = sum(energy_per_point, dims=(1, 2))
    return dropdims(total_energy, dims=(1, 2))
end

function compute_enstrophy_spectra(vorticity)
    nx = size(vorticity, 1)
    ny = size(vorticity, 2)
    kx = fftfreq(nx, nx)
    ky = fftfreq(ny, ny)
    K = (kx.^2)' .+ (ky.^2)  

    K_bins = logrange(1, maximum(K), 100)  

    a = 1.6

    enstrophy_spectrum = zeros(Float32, length(K_bins))

    vorticity_fft = fft(vorticity, [1, 2])
    vorticity_fft_squared = abs2.(vorticity_fft) ./ (prod(size(vorticity_fft))^2)
    for j = 1:length(K_bins)
        bin = K_bins[j]
        mask = (K .> bin / a) .& (K .< bin * a)
        filtered_vorticity = vorticity_fft_squared .* mask
        enstrophy_spectrum[j] = sum(filtered_vorticity)
    end

    return enstrophy_spectrum, K_bins
end

function compute_total_enstrophy(vorticity::CuArray)
    if ndims(vorticity) != 3
        throw(ArgumentError("Input `vorticity` must have 3 dimensions (nx, ny, time_steps). Got: $(size(vorticity))"))
    end
    enstrophy_per_point = abs2.(vorticity)
    total_enstrophy = sum(enstrophy_per_point, dims=(1, 2))
    return dropdims(total_enstrophy, dims=(1, 2))
end

function inference(dt_LES, num_steps, nt, test_data, N_les, stats, velocity_cnn, ps_drift, _st_drift, Re, trajectory_to_evaluate, ϵ, dev; method=:euler_maruyama)
    initial_test_images, test_images, test_labels_closure, test_labels_state = test_data.initial, test_data.target, test_data.closure, test_data.state |> dev;
    initial_test_images, test_images, test_labels_closure, test_labels_state = test_data.initial[:,:,:,:,trajectory_to_evaluate], test_data.target[:,:,:,:,trajectory_to_evaluate], test_data.closure[:,:,:,:,trajectory_to_evaluate], test_data.state[:,:,:,:,trajectory_to_evaluate] |> dev; 
    
    state_means = stats[N_les][:state_means] |> dev;
    state_std = stats[N_les][:state_std] |> dev;
    closure_means = stats[N_les][:closure_means] |> dev;
    closure_std = stats[N_les][:closure_std] |> dev;

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
    global u_les = test_labels_state[:,:,:,1]; 
    global closure = initial_test_images[:,:,:,1];

    u_les = reshape(u_les, N_les, N_les, 2, 1) |> dev; 
    u_les = inverse_standardize_set_per_channel(u_les, state_means, state_std; one_trajectory=true) |> dev;
    u_model = u_les |> dev;
    closure = reshape(closure, N_les, N_les, 2, 1) |> dev;
    closure = inverse_standardize_set_per_channel(closure, closure_means, closure_std; one_trajectory=true) |> dev;
    all_groundtruth = CUDA.zeros(N_les, N_les, 2, nt + 1) |> dev
    all_u_les = CUDA.zeros(N_les, N_les, 2, nt + 1) |> dev
    all_u_model = CUDA.zeros(N_les, N_les, 2, nt + 1) |> dev
    all_groundtruth[:,:,:,1] = u_les |> dev
    all_u_les[:,:,:,1] = u_les |> dev
    all_u_model[:,:,:,1] = u_les |> dev

    non_stand_state = inverse_standardize_set_per_channel(test_labels_state, state_means, state_std; one_trajectory=true) |> dev;

    for i = 2:nt+1
        global u_les, t
        u_les = step_rk4(u_les, dt_LES, f_les) |> dev;
        all_u_les[:,:,:,i] = u_les;
        stand_state_cond = standardize_training_set_per_channel(u_model, state_means, state_std; one_trajectory=true) |> dev;
        stand_closure = standardize_training_set_per_channel(closure, closure_means, closure_std; one_trajectory=true) |> dev;

        if method == :euler_maruyama
            up_stand_closure = generate_closure(velocity_cnn, ps_drift, _st_drift, stand_closure, stand_state_cond, stand_closure, num_steps, ϵ, dev; method=:euler_maruyama);
        elseif method == :heuns_method
            up_stand_closure = generate_closure(velocity_cnn, ps_drift, _st_drift, stand_closure, stand_state_cond, stand_closure, num_steps, ϵ, dev; method=:heuns_method);
        end

        un_stand_closure = inverse_standardize_set_per_channel(up_stand_closure, closure_means, closure_std; one_trajectory=true) |> dev;
        pad_closure = pad_circular(un_stand_closure, 1; dims=1:2)
        projected_closure = INS.project(pad_closure, setup_les; psolver=psolver_les)
        closure = projected_closure[2:end-1, 2:end-1, :, :]
        u_model = step_rk4(u_model, dt_LES, f_les) .+ (dt_LES .* closure) |> dev;

        all_u_model[:,:,:,i] = u_model |> dev;
        all_groundtruth[:,:,:,i] = non_stand_state[:,:,:,i] |> dev
        t += dt_LES
    end
    return all_groundtruth, all_u_les, all_u_model
end

function batched_inference(N_les, batch_size, dt_LES, num_steps, nt, test_data, stats, velocity_cnn, ps_drift, _st_drift, Re, trajectory_to_evaluate, ϵ, dev; method=:euler_maruyama)
    all_groundtruth_batch = CUDA.zeros(Float32, N_les, N_les, 2, nt + 1, batch_size)
    all_u_les_batch = CUDA.zeros(Float32, N_les, N_les, 2, nt + 1, batch_size)
    all_u_model_batch = CUDA.zeros(Float32, N_les, N_les, 2, nt + 1, batch_size)

    for i in 1:batch_size
        if method == :euler_maruyama
            all_groundtruth, all_u_les, all_u_model = inference(dt_LES, num_steps, nt, test_data, N_les, stats, velocity_cnn, ps_drift, _st_drift, Re, trajectory_to_evaluate, ϵ, dev; method=:euler_maruyama)
        elseif method == :heuns_method
            all_groundtruth, all_u_les, all_u_model = inference(dt_LES, num_steps, nt, test_data, N_les, stats, velocity_cnn, ps_drift, _st_drift, Re, trajectory_to_evaluate, ϵ, dev; method=:heuns_method)
        end
        all_groundtruth_batch[:, :, :, :, i] .= all_groundtruth
        all_u_les_batch[:, :, :, :, i] .= all_u_les
        all_u_model_batch[:, :, :, :, i] .= all_u_model
    end

    return all_groundtruth_batch, all_u_les_batch, all_u_model_batch
end

function error_comparison_LES_model(all_groundtruth, all_u_les, all_u_model, dt_LES, save_path)
    nt = size(all_groundtruth, 4) - 1
    batch_size = size(all_groundtruth, 5)
    times = CUDA.collect((0:nt) .* dt_LES)

    mean_diff_les = mean((all_u_les .- all_groundtruth) .^ 2, dims=(1,2,3))
    sum_u_les = sum(all_u_les .^ 2, dims=(1,2,3))
    rrmse_les = mean_diff_les ./ sum_u_les
    rrmse_les = sqrt.(rrmse_les)

    mean_diff_model = mean((all_u_model .- all_groundtruth) .^ 2, dims=(1,2,3))
    sum_u_model = sum(all_u_model .^ 2, dims=(1,2,3))
    rrmse_model = mean_diff_model ./ sum_u_model
    rrmse_model = sqrt.(rrmse_model)

    rrmse_model = reshape(rrmse_model, nt + 1, batch_size)
    rrmse_model_mean = mean(rrmse_model, dims=2)[:, 1]
    rrmse_model_std = std(rrmse_model, dims=2)[:, 1]
    println(" computed the mean: ", rrmse_model_mean, "and std: ", rrmse_model_std)

    rrmse_les = reshape(rrmse_les, nt+1, batch_size)
    rrmse_les = rrmse_les[:,1]

    mse_les_cpu = Array(rrmse_les)
    mse_model_mean_cpu = Array(rrmse_model_mean)
    mse_model_std_cpu = Array(rrmse_model_std)
    times_cpu = Array(times)

    p = plot(times_cpu, mse_les_cpu, label="LES", xlabel="Time", ylabel="RRMSE", lw=2, title="RRMSE Evolution")
    plot!(p, times_cpu, mse_model_mean_cpu .- 3f0 .* mse_model_std_cpu, fillrange=mse_model_mean_cpu .+ 3f0 .* mse_model_std_cpu, color=:green, alpha=0.25, label="")
    plot!(p, times_cpu, mse_model_mean_cpu, label="Model", linewidth=2)
    savefig(p, save_path)
    println("Plot with uncertainty bounds saved to: $save_path")    
end

function plot_total_energy_over_time(all_groundtruth::CuArray, all_u_les::CuArray, all_u_model::CuArray, dt_LES::Float32, save_path::String, batch_size)
    nt = size(all_groundtruth, 4) - 1
    time_vector = CUDA.collect((0:nt) .* dt_LES)

    energy_groundtruth = CUDA.zeros(nt+1, batch_size)
    energy_u_les = CUDA.zeros(nt+1, batch_size)
    energy_u_model = CUDA.zeros(nt+1, batch_size)

    for i in 1:batch_size
        energy_groundtruth[:,i] = vec(compute_total_energy(all_groundtruth[:,:,:,:,i]))
        energy_u_les[:,i] = compute_total_energy(all_u_les[:,:,:,:,i])
        energy_u_model[:,i] = compute_total_energy(all_u_model[:,:,:,:,i])
    end

    energy_model_mean = mean(energy_u_model, dims=2)[:, 1]
    energy_model_std = std(energy_u_model, dims=2)[:, 1]
    println(" computed the mean: ", energy_model_mean, "and std: ", energy_model_std)

    energy_groundtruth_cpu = Array(energy_groundtruth[:,1])
    energy_les_cpu = Array(energy_u_les[:,1])
    energy_model_mean_cpu = Array(energy_model_mean)
    energy_model_std_cpu = Array(energy_model_std)
    time_vector_cpu = Array(time_vector)

    p = plot(time_vector_cpu, energy_groundtruth_cpu, label="Ground Truth", xlabel="Time", ylabel="Energy", lw=2, title="Total Energy")
    plot!(p, time_vector_cpu, energy_les_cpu, label="LES")
    plot!(p, time_vector, energy_model_mean_cpu .- 3f0 .* energy_model_std_cpu, fillrange=energy_model_mean_cpu .+ 3f0 .* energy_model_std_cpu, color=:green, alpha=0.25, label="")
    plot!(p, time_vector_cpu, energy_model_mean_cpu, label="Model", linewidth=2)
    savefig(p, save_path)
    println("Plot with uncertainty bounds saved to: $save_path")    
end

function animation_LES_model_truth(all_u_les_batch, all_u_model_batch, all_groundtruth_batch, nt, dt_LES, N_les, Re, save_path)
    backend = CUDABackend();
    x_les = LinRange(0.0, 1.0, N_les + 1), LinRange(0.0, 1.0, N_les + 1);
    setup_les = INS.Setup(; x=x_les, Re=Re, backend);

    all_u_model = mean(all_u_model_batch, dims=5)[:,:,:,:,1]
    all_groundtruth = all_groundtruth_batch[:,:,:,:,1]
    all_u_les = all_u_les_batch[:,:,:,:,1]
    println(" size of all_u_model: ", size(all_u_model))
    println(" size of all_groundtruth: ", size(all_groundtruth))

    anim = Animation()
    for i in 1:nt
        t = (i - 1) * dt_LES

        ω_les = Array(INS.vorticity(pad_circular(all_u_les[:,:,:,i], 1; dims=1:2), setup_les)) 
        ω_model = Array(INS.vorticity(pad_circular(all_u_model[:,:,:,i], 1; dims=1:2), setup_les)) 
        ω_groundtruth = Array(INS.vorticity(pad_circular(all_groundtruth[:,:,:,i], 1; dims=1:2), setup_les))

        ω_les = ω_les[2:end-1, 2:end-1]
        ω_model = ω_model[2:end-1, 2:end-1]
        ω_groundtruth = ω_groundtruth[2:end-1, 2:end-1]

        all_data = [ω_les, ω_model, ω_groundtruth]
        v_min = minimum([minimum(data) for data in all_data])
        v_max = maximum([maximum(data) for data in all_data])

        p1 = Plots.heatmap(ω_les'; xlabel="x", ylabel="y", title="LES (t=$(round(t, digits=3)))",
                           clim=(v_min, v_max), color=:viridis)
        p2 = Plots.heatmap(ω_model'; xlabel="x", ylabel="y", title="Model (t=$(round(t, digits=3)))",
                           clim=(v_min, v_max), color=:viridis)
        p3 = Plots.heatmap(ω_groundtruth'; xlabel="x", ylabel="y", title="Ground Truth (t=$(round(t, digits=3)))",
                           clim=(v_min, v_max), color=:viridis)

        combined_plot = plot(p1, p2, p3, layout=(1, 3), size=(1800, 600))

        frame(anim, combined_plot)
    end
    gif(anim, save_path, fps=10) 
    println("Animation saved to $save_path")
end

function snapshot_comparison(i, dt_LES, all_groundtruth_batch, all_u_les_batch, all_u_model_batch, N_les, Re, batch_size, nt, save_path; dev)
    t = (i - 1) * dt_LES
    groundtruth = all_groundtruth_batch[:, :, :, i, 1] |> dev
    u_les = all_u_les_batch[:, :, :, i, 1] |> dev
    u_model = all_u_model_batch[:, :, :, i, :] |> dev

    backend = CUDABackend();
    x_les = LinRange(0.0, 1.0, N_les + 1), LinRange(0.0, 1.0, N_les + 1);
    setup_les = INS.Setup(; x=x_les, Re=Re, backend);

    ω_model = CUDA.zeros(N_les, N_les, batch_size);
    for j in 1:batch_size 
        ω_model[:,:,j] = INS.vorticity(pad_circular(u_model[:,:,:,j], 1; dims=1:2), setup_les)[2:end-1, 2:end-1]
    end
    mean_ω_model = mean(ω_model, dims=3)[:,:,1]; 

    ω_nomodel = INS.vorticity(pad_circular(u_les, 1; dims=1:2), setup_les)[2:end-1, 2:end-1]
    ω_groundtruth = INS.vorticity(pad_circular(groundtruth, 1; dims=1:2), setup_les)[2:end-1, 2:end-1]

    mean_ω_model = Array(mean_ω_model)
    ω_nomodel = Array(ω_nomodel) 
    ω_groundtruth = Array(ω_groundtruth) 

    ω_closure = Array(abs.(ω_groundtruth .- ω_nomodel))
    ω_pred_closure = Array(abs.(mean_ω_model .- ω_nomodel)) 
    ω_error_map = Array(abs.(mean_ω_model .- ω_groundtruth)) 

    energy_nomodel, K_bins_energy = compute_energy_spectra(Array(u_les))
    energy_groundtruth, _ = compute_energy_spectra(Array(groundtruth))

    energy_model_batch = CUDA.zeros(length(K_bins_energy), batch_size)

    for j in 1:batch_size
        energy_model, _ = compute_energy_spectra(Array(u_model[:,:,:,j]))
        println(" the size of energy_model: ", size(energy_model[:,1]))
        energy_model_batch[:,j] = energy_model[:,1]
    end

    energy_model_mean = Array(mean(energy_model_batch, dims=2)[:, 1])

    energy_model_std = Array(std(energy_model_batch, dims=2)[:, 1])
    println(" computed the mean: ", energy_model_mean, "and std: ", energy_model_std)

    # enstrophy_nomodel, K_bins_enstrophy = compute_enstrophy_spectra(Array(ω_nomodel))
    # enstrophy_model, _ = compute_enstrophy_spectra(Array(ω_model))
    # enstrophy_groundtruth, _ = compute_enstrophy_spectra(Array(ω_groundtruth))

    title_model = @sprintf("Vorticity model, t = %.3f", t)
    title_nomodel = @sprintf("Vorticity no model, t = %.3f", t)
    title_groundtruth = @sprintf("Vorticity ground truth, t = %.3f", t)
    title_closure = @sprintf("Vorticity closure, t = %.3f", t)
    title_pred_closure = @sprintf("Predicted closure, t = %.3f", t)
    title_error = @sprintf("Error model, t = %.3f", t)

    all_data_state = [mean_ω_model, ω_nomodel, ω_groundtruth]
    all_data_closure = [ω_closure, ω_pred_closure, ω_error_map]
    v_min_state = minimum([minimum(data) for data in all_data_state])
    v_max_state = maximum([maximum(data) for data in all_data_state])
    v_min_closure = minimum([minimum(data) for data in all_data_closure])
    v_max_closure = maximum([maximum(data) for data in all_data_closure])

    p1 = Plots.heatmap(ω_nomodel'; xlabel="x", ylabel="y", title=title_nomodel, color=:viridis, clim=(v_min_state, v_max_state))
    p2 = Plots.heatmap(mean_ω_model'; xlabel="x", ylabel="y", title=title_model, color=:viridis, clim=(v_min_state, v_max_state))
    p3 = Plots.heatmap(ω_groundtruth'; xlabel="x", ylabel="y", title=title_groundtruth, color=:viridis, clim=(v_min_state, v_max_state))
    p4 = Plots.heatmap(ω_closure'; xlabel="x", ylabel="y", title=title_closure, color=:viridis, clim=(v_min_closure, v_max_closure))
    p5 = Plots.heatmap(ω_pred_closure'; xlabel="x", ylabel="y", title=title_pred_closure, color=:viridis, clim=(v_min_closure, v_max_closure))
    p6 = Plots.heatmap(ω_error_map'; xlabel="x", ylabel="y", title=title_error, color=:viridis, clim=(v_min_closure, v_max_closure))

    energy_spectrum_plot = Plots.plot(K_bins_energy, energy_nomodel[:, 1], label="No Model", xaxis=:log, yaxis=:log,
        xlabel="Wavenumber k", ylabel="Energy E(k)",
        title="Energy Spectrum at t=$(round(t, digits=3))")
    Plots.plot!(energy_spectrum_plot, K_bins_energy, energy_groundtruth[:, 1], label="Ground Truth")
    Plots.plot!(energy_spectrum_plot, K_bins_energy, energy_model_mean .- 3f0 .* energy_model_std, fillrange=energy_model_mean .+ 3f0 .* energy_model_std, color=:green, alpha=0.25, label="")
    Plots.plot!(energy_spectrum_plot, K_bins_energy, energy_model_mean, label="Model")

    # enstrophy_spectrum_plot = Plots.plot(K_bins_enstrophy, enstrophy_nomodel[:, 1], label="No Model", xaxis=:log, yaxis=:log,
    #     xlabel="Wavenumber k", ylabel="Enstrophy Z(k)",
    #     title="Enstrophy Spectrum at t=$(round(t, digits=3))")
    # Plots.plot!(enstrophy_spectrum_plot, K_bins_enstrophy, enstrophy_model[:, 1], label="Model")
    # Plots.plot!(enstrophy_spectrum_plot, K_bins_enstrophy, enstrophy_groundtruth[:, 1], label="Ground Truth")

    combined_fig = Plots.plot(p1, p2, p3, p4, p5, p6, energy_spectrum_plot, layout=(3, 3), size=(3200, 2400))
    savefig(combined_fig, joinpath(save_path, @sprintf("time_step_information_%03d.png", i)))

    println("Error between model and ground truth at t = ", t, ": ", mean(ω_error_map))
    println("Error between LES and filtered DNS at t = ", t, ": ", mean(ω_closure))
end

function plot_total_enstrophy_over_time(all_groundtruth, all_u_les, all_u_model, Δt, N_les, Re, save_path)
    num_time_steps = size(all_groundtruth, 4)
    time_vector = collect(0:Δt:(num_time_steps - 1) * Δt)

    backend = CUDABackend()
    x_les = LinRange(0.0, 1.0, N_les + 1), LinRange(0.0, 1.0, N_les + 1)
    setup_les = INS.Setup(; x=x_les, Re=Re, backend)

    vorticity_groundtruth = CUDA.zeros(Float32, size(all_groundtruth, 1), size(all_groundtruth, 2), num_time_steps)
    vorticity_u_les = CUDA.zeros(Float32, size(all_u_les, 1), size(all_u_les, 2), num_time_steps)
    vorticity_u_model = CUDA.zeros(Float32, size(all_u_model, 1), size(all_u_model, 2), num_time_steps)

    for t in 1:num_time_steps
        vorticity_groundtruth[:, :, t] .= INS.vorticity(
            pad_circular(all_groundtruth[:, :, :, t], 1; dims=1:2),
            setup_les
        )[2:end-1, 2:end-1]
        vorticity_u_les[:, :, t] .= INS.vorticity(
            pad_circular(all_u_les[:, :, :, t], 1; dims=1:2),
            setup_les
        )[2:end-1, 2:end-1]
        vorticity_u_model[:, :, t] .= INS.vorticity(
            pad_circular(all_u_model[:, :, :, t], 1; dims=1:2),
            setup_les
        )[2:end-1, 2:end-1]
    end
    println("Vorticity computed for all time steps.")

    total_enstrophy_groundtruth = compute_total_enstrophy(vorticity_groundtruth)
    total_enstrophy_u_les = compute_total_enstrophy(vorticity_u_les)
    total_enstrophy_u_model = compute_total_enstrophy(vorticity_u_model)

    total_enstrophy_groundtruth = Array(total_enstrophy_groundtruth)
    total_enstrophy_u_les = Array(total_enstrophy_u_les)
    total_enstrophy_u_model = Array(total_enstrophy_u_model)

    plt = Plots.plot(
        time_vector,
        total_enstrophy_groundtruth,
        label="Ground Truth",
        xlabel="Time",
        ylabel="Total Enstrophy",
        title="Total Enstrophy Over Time"
    )
    Plots.plot!(plt, time_vector, total_enstrophy_u_les, label="LES")
    Plots.plot!(plt, time_vector, total_enstrophy_u_model, label="Model")

    savefig(plt, save_path)
    println("Plot saved to $save_path")
end








