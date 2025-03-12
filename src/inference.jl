using Plots
using IncompressibleNavierStokes
const INS = IncompressibleNavierStokes
using OrdinaryDiffEq
using FFTW
using KernelAbstractions
using Printf
using StatsPlots

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

function inference(dt_LES, num_steps, nt, test_data, N_les, velocity_cnn, ps_drift, _st_drift, Re, trajectory_to_evaluate, ϵ, dev; method=:euler_maruyama, time_step=:FE)
    initial_test_images, test_images, test_labels_closure, test_labels_state = test_data.initial, test_data.target, test_data.closure, test_data.state |> dev;
    initial_test_images, test_images, test_labels_closure, test_labels_state = test_data.initial[:,:,:,:,trajectory_to_evaluate], test_data.target[:,:,:,:,trajectory_to_evaluate], test_data.closure[:,:,:,:,trajectory_to_evaluate], test_data.state[:,:,:,:,trajectory_to_evaluate] |> dev; 

    create_right_hand_side(setup, psolver) = function right_hand_side(u, p, t)
        u = pad_circular(u, 1; dims = 1:2)
        F = INS.momentum(u, nothing, t, setup)
        F = F[2:end-1, 2:end-1, :]
        F = pad_circular(F, 1; dims = 1:2)
        PF = INS.project(F, setup; psolver)
        PF[2:end-1, 2:end-1, :]
    end

    create_right_hand_side_non_proj(setup) = function right_hand_side(u, p, t)
        u = pad_circular(u, 1; dims = 1:2)
        F = INS.momentum(u, nothing, t, setup)
        F = F[2:end-1, 2:end-1, :]
    end

    backend = CUDABackend();
    x_les = LinRange(0.0, 1.0, N_les + 1), LinRange(0.0, 1.0, N_les + 1);
    setup_les = INS.Setup(; x=x_les, Re=Re, backend);
    psolver_les = INS.psolver_spectral(setup_les);
    f_les = create_right_hand_side(setup_les, psolver_les); 
    f_les_non_proj = create_right_hand_side_non_proj(setup_les);

    global t = 0.0f0; 
    global u_les = test_labels_state[:,:,:,1]; 
    global closure = test_images[:,:,:,1];

    u_les = reshape(u_les, N_les, N_les, 2, 1) |> dev; 
    u_model = u_les |> dev;
    closure = reshape(closure, N_les, N_les, 2, 1) |> dev;
    
    # all_groundtruth = CUDA.zeros(N_les, N_les, 2, nt + 1) |> dev
    # all_u_les = CUDA.zeros(N_les, N_les, 2, nt + 1) |> dev
    all_u_model = CUDA.zeros(N_les, N_les, 2, nt + 1) |> dev
    # all_groundtruth[:,:,:,1] = u_les |> dev
    # all_u_les[:,:,:,1] = u_les |> dev
    all_u_model[:,:,:,1] = u_les |> dev

    non_stand_state = test_labels_state |> dev; 

    for i = 2:nt+1
        global u_les, t
        # u_les = step_rk4(u_les, dt_LES, f_les) |> dev;
        # all_u_les[:,:,:,i] = u_les;
        state_cond = u_model |> dev; 
        closure_cond = closure |> dev;
        if i==2
            proj = f_les_non_proj(state_cond, nothing, 0.0) .+ closure_cond |> dev;
            proj_pad = pad_circular(proj, 1; dims = 1:2) |> dev;
            proj_pad_done = INS.project(proj_pad, setup_les; psolver=psolver_les) |> dev;
            Add = proj_pad_done[2:end-1, 2:end-1, :] |> dev;
            u_model = state_cond .+ (dt_LES .* Add) |> dev; 
        else
            if time_step == :FE
                input_model_c_1 = reshape(closure_cond, N_les, N_les, 2, 1);
                input_model_u_1 = reshape(state_cond, N_les, N_les, 2, 1);
                if method == :euler_maruyama
                    closure = generate_closure(velocity_cnn, ps_drift, _st_drift, input_model_c_1, input_model_u_1, input_model_c_1, num_steps, ϵ, dev; method=:euler_maruyama) |> dev;
                elseif method == :heuns_method
                    closure = generate_closure(velocity_cnn, ps_drift, _st_drift, input_model_c_1, input_model_u_1, input_model_c_1, num_steps, ϵ, dev; method=:heuns_method) |> dev;                
                end
                proj = f_les_non_proj(state_cond, nothing, 0.0) .+ closure |> dev;
                proj_pad = pad_circular(proj, 1; dims = 1:2) |> dev;
                proj_pad_done = INS.project(proj_pad, setup_les; psolver=psolver_les) |> dev;
                Add = proj_pad_done[2:end-1, 2:end-1, :] |> dev;
                u_model = state_cond .+ (dt_LES .* Add) |> dev; 
            elseif time_step == :RK4
                if method == :euler_maruyama
                    u_model, closure = step_rk4_with_closure(state_cond, closure_cond, dt_LES, velocity_cnn, ps_drift, _st_drift, num_steps, ϵ, N_les, f_les_non_proj, setup_les, psolver_les; dev, method=:euler_maruyama)
                elseif method == :heuns_method
                    u_model, closure = step_rk4_with_closure(state_cond, closure_cond, dt_LES, velocity_cnn, ps_drift, _st_drift, num_steps, ϵ, N_les, f_les_non_proj, setup_les, psolver_les; dev, method=:heuns_method)                
                end
            end
            pad_closure = pad_circular(closure, 1; dims = 1:2) |> dev;
            proj_pad_closure = INS.project(pad_closure, setup_les; psolver=psolver_les) |> dev;
            closure = proj_pad_closure[2:end-1, 2:end-1, :] |> dev;
        end
        all_u_model[:,:,:,i] = u_model |> dev;
        # all_groundtruth[:,:,:,i] = non_stand_state[:,:,:,i] |> dev
        t += dt_LES
    end
    return all_u_model # all_groundtruth, all_u_les, all_u_model
end

function inference_deterministic(dt_LES, nt, test_data, N_les, velocity_cnn_det, ps_deterministic, _st_deterministic, Re, trajectory_to_evaluate, dev)
    test_images, test_labels_state = test_data.target, test_data.state |> dev;
    test_images, test_labels_state = test_data.target[:,:,:,:,trajectory_to_evaluate], test_data.state[:,:,:,:,trajectory_to_evaluate] |> dev; 

    create_right_hand_side(setup) = function right_hand_side(u, p, t)
        u = pad_circular(u, 1; dims = 1:2)
        F = INS.momentum(u, nothing, t, setup)
        F = F[2:end-1, 2:end-1, :]
    end

    backend = CUDABackend();
    x_les = LinRange(0.0, 1.0, N_les + 1), LinRange(0.0, 1.0, N_les + 1);
    setup_les = INS.Setup(; x=x_les, Re=Re, backend);
    psolver_les = INS.psolver_spectral(setup_les);
    f_les = create_right_hand_side(setup_les); 

    global t = 0.0f0; 
    global u_model = test_labels_state[:,:,:,1]; 

    u_model = reshape(u_model, N_les, N_les, 2, 1) |> dev; 
    all_u_model = CUDA.zeros(N_les, N_les, 2, nt + 1) |> dev;
    all_u_model[:,:,:,1] = u_model
    for i = 2:nt+1
        global t
        state_cond = u_model |> dev;
        u_model = step_rk4_with_closure_deterministic(state_cond, dt_LES, velocity_cnn_det, ps_deterministic, _st_deterministic, N_les, f_les, setup_les, psolver_les; dev)
        all_u_model[:,:,:,i] = u_model |> dev;
        t += dt_LES
    end
    return all_u_model
end

function batched_inference(N_les, batch_size, dt_LES, num_steps, nt, test_data, velocity_cnn, velocity_cnn_det, ps_drift, _st_drift, ps_deterministic, _st_deterministic, Re, trajectory_to_evaluate, ϵ, dev; method=:euler_maruyama, time_step=:FE)
    # all_groundtruth_batch = CUDA.zeros(Float32, N_les, N_les, 2, nt + 1, batch_size)
    # all_u_les_batch = CUDA.zeros(Float32, N_les, N_les, 2, nt + 1, batch_size)
    all_u_model_batch = CUDA.zeros(Float32, N_les, N_les, 2, nt + 1, batch_size)
    all_u_deterministic_batch = CUDA.zeros(Float32, N_les, N_les, 2, nt+1, batch_size)

    total_time_si = 0.0
    total_time_det = 0.0

    for i in 1:batch_size
        if time_step == :FE
            comp_time_si = @elapsed begin
                if method == :euler_maruyama
                    all_u_model = inference(dt_LES, num_steps, nt, test_data, N_les, velocity_cnn, ps_drift, _st_drift, Re, trajectory_to_evaluate, ϵ, dev; method=:euler_maruyama, time_step=:FE) 
                elseif method == :heuns_method
                    all_u_model = inference(dt_LES, num_steps, nt, test_data, N_les, velocity_cnn, ps_drift, _st_drift, Re, trajectory_to_evaluate, ϵ, dev; method=:heuns_method, time_step=:FE)
                end
            end
        elseif time_step == :RK4
            comp_time_si = @elapsed begin
                if method == :euler_maruyama
                    all_u_model = inference(dt_LES, num_steps, nt, test_data, N_les, velocity_cnn, ps_drift, _st_drift, Re, trajectory_to_evaluate, ϵ, dev; method=:euler_maruyama, time_step=:RK4) 
                elseif method == :heuns_method
                    all_u_model = inference(dt_LES, num_steps, nt, test_data, N_les, velocity_cnn, ps_drift, _st_drift, Re, trajectory_to_evaluate, ϵ, dev; method=:heuns_method, time_step=:RK4)
                end
            end
        end
        comp_time_det = @elapsed begin
            all_deterministic = inference_deterministic(dt_LES, nt, test_data, N_les, velocity_cnn_det, ps_deterministic, _st_deterministic, Re, trajectory_to_evaluate, dev)
        end
        total_time_si += comp_time_si
        total_time_det += comp_time_det
        # all_groundtruth_batch[:, :, :, :, i] .= all_groundtruth
        # all_u_les_batch[:, :, :, :, i] .= all_u_les
        all_u_model_batch[:, :, :, :, i] .= all_u_model
        all_u_deterministic_batch[:, :, :, :, i] .= all_deterministic
    end

    # Compute and print the average times
    avg_time_si = total_time_si / batch_size
    avg_time_det = total_time_det / batch_size

    println("Total computation time for stochastic inference (SI): $(total_time_si) seconds")
    println("Average computation time per batch for SI: $(avg_time_si) seconds")

    println("Total computation time for deterministic inference: $(total_time_det) seconds")
    println("Average computation time per batch for deterministic inference: $(avg_time_det) seconds")


    return all_groundtruth_batch, all_u_les_batch, all_u_model_batch, all_u_deterministic_batch
end

function compute_aposteriori_error_gpu(error_matrix) 
    n_t = size(error_matrix, 1)  

    cumulative_sum = cumsum(error_matrix, dims=1)  
    indices = CuArray(1:n_t)  
    indices = reshape(indices, n_t, 1) 

    aposteriori_error = cumulative_sum ./ indices  
    return aposteriori_error
end

function error_comparison_LES_model(all_groundtruth, all_u_les, all_u_model, all_u_deterministic, dt_LES, save_path; dev)
    nt = size(all_groundtruth, 4) - 1
    batch_size = size(all_groundtruth, 5)
    times = CUDA.collect((0:nt) .* dt_LES)
    rrmse_model = CUDA.zeros(Float32, nt+1, batch_size)
    for i=1:batch_size
        y_pred = all_u_model[:,:,:,:,i] |> dev;
        y_true = all_groundtruth[:,:,:,:,1] |> dev;
        for j=1:nt+1
            y = y_pred[:,:,:,j] |> dev;
            y_hat = y_true[:,:,:,j] |> dev;
            rrmse = prior(y_hat, y; dev) |> dev;
            CUDA.@allowscalar rrmse_model[j,i] = rrmse;
        end
    end
    # rrmse_post = compute_aposteriori_error_gpu(rrmse_model) |> dev; 

    rrmse_model_cpu = Array(rrmse_model) # Array(rrmse_post)
    median_rrmse_cpu = median(rrmse_model_cpu, dims=2)[:, 1]
    min_rrmse_cpu = minimum(rrmse_model_cpu, dims=2)[:, 1]
    max_rrmse_cpu = maximum(rrmse_model_cpu, dims=2)[:, 1]
    q1_rrmse = [quantile(rrmse_model_cpu[t, :], 0.25) for t in 1:(nt+1)] 
    q3_rrmse = [quantile(rrmse_model_cpu[t, :], 0.75) for t in 1:(nt+1)] 
    q1_rrmse = collect(q1_rrmse)
    q3_rrmse = collect(q3_rrmse)

    rrmse_no_model = CUDA.zeros(nt+1)
    for j=1:nt+1
        y = all_u_les[:,:,:,j,1] |> dev;
        y_hat = all_groundtruth[:,:,:,j,1] |> dev;
        rrmse = relative_root_mse(y_hat, y; dev) |> dev;
        CUDA.@allowscalar rrmse_no_model[j] = rrmse |> dev;
    end
    # rrmse_no_post = compute_aposteriori_error_gpu(rrmse_no_model) |> dev; 
    mse_les_cpu = Array(rrmse_no_model) # Array(rrmse_no_post)

    rrmse_det_model = CUDA.zeros(nt+1)
    for j=1:nt+1
        y = all_u_deterministic[:,:,:,j,1] |> dev;
        y_hat = all_groundtruth[:,:,:,j,1] |> dev;
        rrmse = relative_root_mse(y_hat, y; dev) |> dev;
        CUDA.@allowscalar rrmse_det_model[j] = rrmse |> dev;
    end
    # rrmse_det_post = compute_aposteriori_error_gpu(rrmse_det_model) |> dev; 
    mse_det_model_cpu = Array(rrmse_det_model) # Array(rrmse_det_post)
    times_cpu = Array(times)

    p = plot(times_cpu, mse_les_cpu, label="No Model", xlabel="Time", ylabel="RMSE", lw=2, title="Error Evolution", color=:green)
    plot!(p, times_cpu, mse_det_model_cpu, label="Det. Model", lw=2, color=:red)
    plot!(times_cpu, median_rrmse_cpu, label="Median SI Model", lw=2, color=:blue)
    plot!(times_cpu, q1_rrmse, ribbon=(q3_rrmse - q1_rrmse), fillalpha=0.3, label="IQR (Q1-Q3)", lw=0, color=:blue)
    plot!(times_cpu, min_rrmse_cpu, ribbon=(max_rrmse_cpu - min_rrmse_cpu), fillalpha=0.2, label="Min/Max Range", lw=0, color=:grey)
    savefig(p, save_path)
    println("Plot with uncertainty bounds saved to: $save_path")    
end

function probability_density_error(all_u_model, all_groundtruth, all_u_deterministic, t, batch_size, nt; method=:euler_maruyama, dev)
    rrmse_model = CUDA.zeros(nt, batch_size)
    for i=1:batch_size
        y_pred = all_u_model[:,:,:,:,i] |> dev;
        y_true = all_groundtruth[:,:,:,:,1] |> dev;
        for j=1:nt
            y = y_pred[:,:,:,j] |> dev;
            y_hat = y_true[:,:,:,j] |> dev;
            rrmse = prior(y_hat, y; dev) |> dev;
            CUDA.@allowscalar rrmse_model[j,i] = rrmse;
        end
    end
    # rrmse_post = compute_aposteriori_error_gpu(rrmse_model) |> dev; 
    error_u_model_cpu = Array(rrmse_model[nt,:])

    rrmse_det_model = CUDA.zeros(nt)
    for j=1:nt
        y = all_u_deterministic[:,:,:,j,1] |> dev;
        y_hat = all_groundtruth[:,:,:,j,1] |> dev;
        rrmse = relative_root_mse(y_hat, y; dev) |> dev;
        CUDA.@allowscalar rrmse_det_model[j] = rrmse |> dev;
    end
    # rrmse_det_post = compute_aposteriori_error_gpu(rrmse_det_model) |> dev; 
    error_u_det_model_cpu = Array(rrmse_det_model)[nt]

    p = density(error_u_model_cpu, xlabel="RMSE", ylabel="Density", 
                title="Density Plot of the RMSE at Time $t", label="SI",
                color=:blue, fillalpha=0.3, ylims=(0,1)) 
    vline!([error_u_det_model_cpu], color=:red, label="Det Model")
    savefig(p, "figures/density_total_error_$nt$method.png")
end


function plot_total_energy_over_time(all_groundtruth::CuArray, all_u_les::CuArray, all_u_model::CuArray, all_u_deterministic::CuArray, dt_LES::Float32, save_path::String, batch_size)
    nt = size(all_groundtruth, 4) - 1
    time_vector = CUDA.collect((0:nt) .* dt_LES)

    energy_groundtruth = CUDA.zeros(nt+1, batch_size)
    energy_u_les = CUDA.zeros(nt+1, batch_size)
    energy_u_model = CUDA.zeros(nt+1, batch_size)
    energy_u_deterministic = CUDA.zeros(nt+1, batch_size)

    for i in 1:batch_size
        energy_groundtruth[:,i] = vec(compute_total_energy(all_groundtruth[:,:,:,:,i]))
        energy_u_les[:,i] = compute_total_energy(all_u_les[:,:,:,:,i])
        energy_u_model[:,i] = compute_total_energy(all_u_model[:,:,:,:,i])
        energy_u_deterministic[:,i] = compute_total_energy(all_u_deterministic[:,:,:,:,i])
    end
    energy_u_model_cpu = Array(energy_u_model)
    median_model_cpu = median(energy_u_model_cpu, dims=2)[:, 1]
    min_model_cpu = minimum(energy_u_model_cpu, dims=2)[:, 1]
    max_model_cpu = maximum(energy_u_model_cpu, dims=2)[:, 1]
    q1_model = [quantile(energy_u_model_cpu[t, :], 0.25) for t in 1:(nt+1)]  
    q3_model = [quantile(energy_u_model_cpu[t, :], 0.75) for t in 1:(nt+1)]
    q1_model = collect(q1_model)
    q3_model = collect(q3_model)

    energy_groundtruth_cpu = Array(energy_groundtruth[:,1])
    energy_les_cpu = Array(energy_u_les[:,1])
    energy_deterministic_model_cpu = Array(energy_u_deterministic[:,1])
    time_vector_cpu = Array(time_vector)

    p = plot(time_vector_cpu, energy_groundtruth_cpu, label="Ground Truth", xlabel="Time", ylabel="Energy", lw=2, title="Total Energy", color=:orange)
    plot!(p, time_vector_cpu, energy_deterministic_model_cpu, label="Det model", color=:red)
    plot!(p, time_vector_cpu, energy_les_cpu, label="No model", color=:green)
    plot!(time_vector_cpu, median_model_cpu, label="Median SI Model", lw=2, color=:blue)
    plot!(time_vector_cpu, q1_model, ribbon=(q3_model - q1_model), fillalpha=0.3, label="IQR (Q1-Q3)", lw=0, color=:blue)
    plot!(time_vector_cpu, min_model_cpu, ribbon=(max_model_cpu - min_model_cpu), fillalpha=0.2, label="Min/Max Range", lw=0, color=:grey)
    savefig(p, save_path)
    println("Plot with uncertainty bounds saved to: $save_path")    
end

function probability_density_energy(all_u_model, all_groundtruth, all_u_deterministic, t, batch_size, time; method=:euler_maruyama)
    nt = size(all_groundtruth, 4) - 1
    energy_groundtruth = CUDA.zeros(nt+1, batch_size)
    energy_u_model = CUDA.zeros(nt+1, batch_size)
    energy_u_deterministic = CUDA.zeros(nt+1, batch_size)

    for i in 1:batch_size
        energy_groundtruth[:,i] = vec(compute_total_energy(all_groundtruth[:,:,:,:,i]))
        energy_u_model[:,i] = compute_total_energy(all_u_model[:,:,:,:,i])
        energy_u_deterministic[:,i] = compute_total_energy(all_u_deterministic[:,:,:,:,i])
    end
    energy_u_model_cpu = Array(energy_u_model)[time,:]
    energy_groundtruth_cpu = Array(energy_groundtruth)[time, 1]
    energy_u_deterministic_cpu = Array(energy_u_deterministic)[time, 1]

    p = density(energy_u_model_cpu, xlabel="Total Energy", ylabel="Density", 
                title="Density Plot of the total energy at time $t", label="SI",
                color=:blue, fillalpha=0.3, ylims=(0,1)) 
    vline!([energy_groundtruth_cpu], color=:green, label="Det Model")
    vline!([energy_u_deterministic_cpu], color=:red, label="Ground Truth")
    savefig(p, "figures/density_total_energy_$nt$method.png")
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

function plot_energy_spectrums(i, dt_LES, all_groundtruth_batch, all_u_les_batch, all_u_model_batch, all_u_det_batch, save_path, batch_size; dev)
    t = (i - 1) * dt_LES
    groundtruth = all_groundtruth_batch[:, :, :, i, 1] |> dev
    u_les = all_u_les_batch[:, :, :, i, 1] |> dev
    u_model = all_u_model_batch[:, :, :, i, :] |> dev
    u_det_model = all_u_det_batch[:, :, :, i, 1] |> dev

    energy_nomodel, K_bins_energy = compute_energy_spectra(Array(u_les))
    energy_groundtruth, _ = compute_energy_spectra(Array(groundtruth))
    energy_det_model, _ = compute_energy_spectra(Array(u_det_model))

    energy_model_batch = CUDA.zeros(length(K_bins_energy), batch_size)

    for j in 1:batch_size
        energy_model, _ = compute_energy_spectra(Array(u_model[:,:,:,j]))
        energy_model_batch[:,j] = energy_model[:,1]
    end
    energy_model_batch_cpu = Array(energy_model_batch)
    median_model_cpu = median(energy_model_batch_cpu, dims=2)[:, 1]
    min_model_cpu = minimum(energy_model_batch_cpu, dims=2)[:, 1]
    max_model_cpu = maximum(energy_model_batch_cpu, dims=2)[:, 1]
    q1_model = [quantile(energy_model_batch_cpu[t, :], 0.25) for t in 1:length(K_bins_energy)]  
    q3_model = [quantile(energy_model_batch_cpu[t, :], 0.75) for t in 1:length(K_bins_energy)]
    q1_model = collect(q1_model)
    q3_model = collect(q3_model)

    energy_spectrum_plot = Plots.plot(K_bins_energy, energy_nomodel[:, 1], label="No Model", xaxis=:log, yaxis=:log, xlabel="κ", ylabel="Energy E(κ)", title="Energy Spectrum at t=$(round(t, digits=3))", color=:green)
    plot!(energy_spectrum_plot, K_bins_energy, energy_groundtruth[:, 1], label="Ground Truth", color=:black)
    plot!(energy_spectrum_plot, K_bins_energy, median_model_cpu, label="Median SI Model", lw=2, color=:blue)
    plot!(energy_spectrum_plot, K_bins_energy, q1_model, ribbon=(q3_model - q1_model), fillalpha=0.3, label="IQR (Q1-Q3)", lw=0, color=:blue)
    plot!(energy_spectrum_plot, K_bins_energy, min_model_cpu, ribbon=(max_model_cpu - min_model_cpu), fillalpha=0.2, label="Min/Max Range", lw=0, color=:grey)
    plot!(energy_spectrum_plot, K_bins_energy, energy_det_model[:,1], label="Det Model", color=:red)

    mkpath(save_path)
    save_filename = joinpath(save_path, "Heun_energy_spectrum_t$(i).png")
    savefig(energy_spectrum_plot, save_filename)
    println("Saved energy spectrum plot: $save_filename")
end


function snapshot_comparison(i, dt_LES, all_groundtruth_batch, all_u_les_batch, all_u_model_batch, all_u_det_model, N_les, Re, batch_size, save_path; dev)
    t = (i - 1) * dt_LES
    groundtruth = all_groundtruth_batch[:, :, :, i, 1] |> dev
    u_les = all_u_les_batch[:, :, :, i, 1] |> dev
    u_model = all_u_model_batch[:, :, :, i, :] |> dev
    u_det_model = all_u_det_model[:, :, :, i, 1] |> dev

    backend = CUDABackend();
    x_les = LinRange(0.0, 1.0, N_les + 1), LinRange(0.0, 1.0, N_les + 1);
    setup_les = INS.Setup(; x=x_les, Re=Re, backend);

    ω_model = CUDA.zeros(N_les, N_les, batch_size);
    for j in 1:batch_size 
        ω_model[:,:,j] = INS.vorticity(pad_circular(u_model[:,:,:,j], 1; dims=1:2), setup_les)[2:end-1, 2:end-1]
    end
    all_vorticities = [Array(ω_model[:,:,j]) for j in 1:batch_size]
    v_min = minimum([minimum(v) for v in all_vorticities])
    v_max = maximum([maximum(v) for v in all_vorticities])
    vorticity_plots = [Plots.heatmap(v'; xlabel="x", ylabel="y", title=@sprintf("Model Vorticity Sample %d", j), color=:viridis, clim=(v_min, v_max)) for (j, v) in enumerate(all_vorticities)]
    combined_vorticity_plot = Plots.plot(vorticity_plots..., layout=(ceil(Int, batch_size / 3), 3), size=(3200, 3200), aspect_ratio=:equal)
    savefig(combined_vorticity_plot, joinpath(save_path, @sprintf("vorticity_batch_timestep_%03d_Heun.png", i)))

    mean_ω_model = mean(ω_model, dims=3)[:,:,1]; 
    std_ω_model = std(ω_model, dims=3)[:,:,1]; 

    ω_nomodel = INS.vorticity(pad_circular(u_les, 1; dims=1:2), setup_les)[2:end-1, 2:end-1]
    ω_groundtruth = INS.vorticity(pad_circular(groundtruth, 1; dims=1:2), setup_les)[2:end-1, 2:end-1]
    ω_det_model = INS.vorticity(pad_circular(u_det_model, 1; dims=1:2), setup_les)[2:end-1, 2:end-1]

    std_ω_model_cpu = Array(std_ω_model)
    mean_ω_model_cpu = Array(mean_ω_model)
    ω_nomodel_cpu = Array(ω_nomodel) 
    ω_groundtruth_cpu = Array(ω_groundtruth) 
    ω_det_model_cpu = Array(ω_det_model)

    ω_error_no_model_cpu = Array(abs.(ω_groundtruth .- ω_nomodel))
    ω_error_det_model_cpu = Array(abs.(ω_groundtruth .- ω_det_model))
    ω_error_model = CUDA.zeros(N_les, N_les, batch_size);
    for j in 1:batch_size 
        ω_error_model[:,:,j] .= abs.(ω_groundtruth .- ω_model[:,:,j])
    end
    mean_ω_error_model = mean(ω_error_model, dims=3)[:,:,1]; 
    std_ω_error_model = std(ω_error_model, dims=3)[:,:,1]; 
    mean_ω_error_model_cpu = Array(mean_ω_error_model)
    std_ω_error_model_cpu = Array(std_ω_error_model)

    all_errors = [Array(ω_error_model[:,:,j]) for j in 1:batch_size]
    v_min_error = minimum([minimum(e) for e in all_errors])
    v_max_error = maximum([maximum(e) for e in all_errors])
    error_plots = [Plots.heatmap(e'; xlabel="x", ylabel="y", title=@sprintf("Model Error Sample %d", j), color=:viridis, clim=(v_min_error, v_max_error)) for (j, e) in enumerate(all_errors)]
    combined_error_plot = Plots.plot(error_plots..., layout=(ceil(Int, batch_size / 3), 3), size=(3200, 3200), aspect_ratio=:equal)
    savefig(combined_error_plot, joinpath(save_path, @sprintf("error_batch_timestep_%03d_Heun.png", i)))

    title_mean_model = @sprintf("Mean SI Model, t = %.3f", t)
    title_std_model = @sprintf("Std SI Model, t=%.3f", t)
    title_det_model = @sprintf("Det model, t=%.3f", t)
    title_nomodel = @sprintf("No Model, t = %.3f", t)
    title_groundtruth = @sprintf("Ground Truth, t = %.3f", t)
    title_error_mean_model = @sprintf("Mean Error SI Model, t=%.3f", t)
    title_error_std_model = @sprintf("Std Error SI Model, t=%.3f", t)
    title_error_det_model = @sprintf("Error Det Model, t=%.3f", t)
    title_error_no_model = @sprintf("Error No Model, t=%.3f", t)

    all_data_state = [mean_ω_model_cpu, ω_nomodel_cpu, ω_groundtruth_cpu, ω_det_model_cpu]
    all_data_error = [ω_error_no_model_cpu, ω_error_det_model_cpu, mean_ω_error_model_cpu]
    v_min_state = minimum([minimum(data) for data in all_data_state])
    v_max_state = maximum([maximum(data) for data in all_data_state])
    v_min_error = minimum([minimum(data) for data in all_data_error])
    v_max_error = maximum([maximum(data) for data in all_data_error])

    p1 = Plots.heatmap(mean_ω_model_cpu'; title=title_mean_model, color=:viridis, clim=(v_min_state, v_max_state))
    p2 = Plots.heatmap(std_ω_model_cpu'; title=title_std_model, color=:plasma)
    p3 = Plots.heatmap(ω_det_model_cpu'; title=title_det_model, color=:viridis, clim=(v_min_state, v_max_state))
    p4 = Plots.heatmap(ω_nomodel_cpu'; title=title_nomodel, color=:viridis, clim=(v_min_state, v_max_state))
    p5 = Plots.heatmap(ω_groundtruth_cpu'; title=title_groundtruth, color=:viridis, clim=(v_min_state, v_max_state))

    p6 = Plots.heatmap(mean_ω_error_model_cpu'; title=title_error_mean_model, color=:viridis, clim=(v_min_error, v_max_error))
    p7 = Plots.heatmap(std_ω_error_model_cpu'; title=title_error_std_model, color=:plasma)
    p8 = Plots.heatmap(ω_error_det_model_cpu'; title=title_error_det_model, color=:viridis, clim=(v_min_error, v_max_error))
    p9 = Plots.heatmap(ω_error_no_model_cpu'; title=title_error_no_model, color=:viridis, clim=(v_min_error, v_max_error))

    combined_fig = Plots.plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, layout=(3, 3), size=(3200, 3200), aspect_ratio=:equal)
    savefig(combined_fig, joinpath(save_path, @sprintf("time_step_information_%03d_Heun.png", i)))
end

function plot_batch(N_les, batch_size, all_u_model_batch, Re, i; dev)
    u_model = all_u_model_batch[:, :, :, i, :] |> dev

    backend = CUDABackend();
    x_les = LinRange(0.0, 1.0, N_les + 1), LinRange(0.0, 1.0, N_les + 1);
    setup_les = INS.Setup(; x=x_les, Re=Re, backend);

    ω_model = CUDA.zeros(N_les, N_les, batch_size);
    for j in 1:batch_size 
        ω_model[:,:,j] = INS.vorticity(pad_circular(u_model[:,:,:,j], 1; dims=1:2), setup_les)[2:end-1, 2:end-1]
    end
    ω_model_cpu = Array(ω_model)
    for j in 1:batch_size
        heatmap(ω_model_cpu[:,:,j], title="Vorticity Batch $j", color=:viridis)
        savefig("vorticity_batch_$j.png")
    end
end






