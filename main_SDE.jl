using master_project

using Lux
using Random
using ComponentArrays
using Optimisers
using Statistics
using Plots
using LuxCUDA
using Printf
using CUDA
using Serialization
using FileIO
using IncompressibleNavierStokes
const INS = IncompressibleNavierStokes
using OrdinaryDiffEq
using LinearAlgebra

dev = gpu_device()
if dev==gpu_device()
    gr()
    ENV[" GKSwstype"] = "100"
    CUDA.allowscalar(false)
end

####### DATA GENERATION ####### 
output_dir = "datasets/trajectories";
generate_new_data = false;
Re = 1.0f4;
num_trajectories = 32;
N_les = 64; 
N_dns = 2048; 
dt = 1.0f-4;
dt_LES = 10 * dt; 
nt = 100000; 
ϵ = 0.1;

data_v, data_c = generate_or_load_data(N_dns, N_les, Re, output_dir, generate_new_data, nt, dt, num_trajectories; dev) |> dev;
initial_sample, target_sample, target_label_closure, target_label_state = create_training_sets(data_c, data_v, N_les);
splits = split_trajectories(initial_sample, target_sample, target_label_closure, target_label_state); 

train_data = splits.train
val_data = splits.val
test_data = splits.test

####### INITIALIZE/LOAD STOCHASTIC INTERPOLANT MODEL ####### 
velocity_cnn =  build_full_unet(16,[32,64,128,256],128; dev) #build_full_unet(8,[8,16,32,64],128; dev)
model_name = "big_closure_to_closure_explicit";
load_path = "trained_models/$model_name.bson";
ps_drift, st_drift, opt_drift = initialize_or_load_model(model_name, velocity_cnn, load_path; dev) |> dev;

# num_params =  Lux.parameterlength(ps_drift)
# println("Total number of parameters: ", num_params)

# ####### TRAINING STOCHASTIC INTERPOLANT #######
# num_training_time_steps = 5000; 
# batch_size = 8;
# num_epochs = 20;
# train!(train_data, val_data, batch_size, num_epochs, ps_drift, st_drift, opt_drift, velocity_cnn, "trained_models", model_name, num_training_time_steps, load_path, ϵ; dev);

####### INITIALIZE/LOAD MODEL DETERMINISTIC MODEL ####### 
velocity_cnn_det = build_full_unet_det(16,[32,64,128,256]) |> dev; 
model_name_det = "big_deterministic";
load_path_det = "trained_models/$model_name_det.bson";
ps_deterministic, st_deterministic, opt_deterministic = initialize_or_load_model(model_name_det, velocity_cnn_det, load_path_det; dev) |> dev;

# ####### TRAINING #######
# num_training_time_steps = 5000; 
# batch_size = 32;
# num_epochs = 50;
# train_deterministic!(train_data, val_data, batch_size, num_epochs, ps_deterministic, st_deterministic, opt_deterministic, velocity_cnn_det, "trained_models", model_name_det, num_training_time_steps, load_path_det; dev);

####### EVALUATION CLOSURE #######
_st_drift = Lux.testmode(st_drift) |> dev;
_st_deterministic = Lux.testmode(st_deterministic) |> dev;

# backend = CUDABackend();
# x_les = LinRange(0.0, 1.0, N_les + 1), LinRange(0.0, 1.0, N_les + 1);
# setup_les = INS.Setup(; x=x_les, Re=Re, backend);
# psolver_les = INS.psolver_spectral(setup_les);

# time_steps_dependency(test_data, ϵ; dev, method=:euler_maruyama)
# time_steps_dependency(test_data, ϵ; dev, method=:heuns_method)

####### EVALUATION STATE/CLOSURE #######
nt = 200;
num_steps = 50;
trajectory_to_evaluate = 1;
batch_size = 10;

all_groundtruth_batch, all_u_les_batch, all_u_model_batch, all_u_deterministic_batch, closure_SI_batch, closure_groundtruth_batch, closure_det_batch = batched_inference(N_les, batch_size, dt_LES, num_steps, nt, train_data, velocity_cnn, velocity_cnn_det, ps_drift, _st_drift, ps_deterministic, _st_deterministic, Re, trajectory_to_evaluate, ϵ, dev; method=:heuns_method)

# function compute_mean_norms(data_batches, time_steps)
#     mean_norms = Dict()
    
#     for nt in time_steps
#         mean_norms[nt] = Dict()
        
#         for (name, data) in data_batches
#             # Compute the Frobenius norm at the given time step
#             norm_values = [norm(data[:, :, :, nt, b]) for b in 1:size(data, 5)]
            
#             # Compute the mean over the batch
#             mean_norms[nt][name] = mean(norm_values)
#         end
#     end
    
#     return mean_norms
# end

# # Example usage:
# time_steps = [1, 51, 101, 151, 201]

# data_batches = Dict(
#     "all_groundtruth_batch" => all_groundtruth_batch,
#     "all_u_les_batch" => all_u_les_batch,
#     "all_u_model_batch" => all_u_model_batch,
#     "all_u_deterministic_batch" => all_u_deterministic_batch,
#     "closure_SI_batch" => closure_SI_batch,
#     "closure_groundtruth_batch" => closure_groundtruth_batch,
#     "closure_det_batch" => closure_det_batch
# )

# mean_norms = compute_mean_norms(data_batches, time_steps)

# # Print the results
# for nt in time_steps
#     println("Time step nt=$nt:")
#     for (name, value) in mean_norms[nt]
#         println("  $name: $value")
#     end
# end


# error_comparison_LES_model_closure(closure_groundtruth_batch, closure_SI_batch, closure_det_batch, dt_LES, "priori_error_non_proj.png"; dev)
# error_comparison_LES_model(all_groundtruth_batch, all_u_les_batch, all_u_model_batch, all_u_deterministic_batch, dt_LES, "posteriori_error_non_proj.png"; dev)

plot_energy_spectrums(51, dt_LES, all_groundtruth_batch, all_u_les_batch, all_u_model_batch, all_u_deterministic_batch, "figures", batch_size; dev)
plot_energy_spectrums(101, dt_LES, all_groundtruth_batch, all_u_les_batch, all_u_model_batch, all_u_deterministic_batch, "figures", batch_size; dev)
plot_energy_spectrums(151, dt_LES, all_groundtruth_batch, all_u_les_batch, all_u_model_batch, all_u_deterministic_batch, "figures", batch_size; dev)
plot_energy_spectrums(201, dt_LES, all_groundtruth_batch, all_u_les_batch, all_u_model_batch, all_u_deterministic_batch, "figures", batch_size; dev)
# plot_total_energy_over_time(all_groundtruth_batch, all_u_les_batch, all_u_model_batch, all_u_deterministic_batch, dt_LES, "total_energy.png", batch_size)
# snapshot_comparison(51, dt_LES, all_groundtruth_batch, all_u_les_batch, all_u_model_batch, all_u_deterministic_batch, N_les, Re, batch_size, "figures"; dev)
# snapshot_comparison(101, dt_LES, all_groundtruth_batch, all_u_les_batch, all_u_model_batch, all_u_deterministic_batch, N_les, Re, batch_size, "figures"; dev)
# snapshot_comparison(151, dt_LES, all_groundtruth_batch, all_u_les_batch, all_u_model_batch, all_u_deterministic_batch, N_les, Re, batch_size, "figures"; dev)
# snapshot_comparison(201, dt_LES, all_groundtruth_batch, all_u_les_batch, all_u_model_batch, all_u_deterministic_batch, N_les, Re, batch_size, "figures"; dev)


# probability_density_error(all_u_model_batch, all_groundtruth_batch, all_u_deterministic_batch, 0.05, batch_size, 51; method=:euler_maruyama, dev)
# probability_density_error(all_u_model_batch, all_groundtruth_batch, all_u_deterministic_batch, 0.1, batch_size, 101; method=:euler_maruyama, dev)
# # # probability_density_error(all_u_model_batch, all_groundtruth_batch, all_u_deterministic_batch, 0.15, batch_size, 151; method=:euler_maruyama, dev)
# # # probability_density_error(all_u_model_batch, all_groundtruth_batch, all_u_deterministic_batch, 0.2, batch_size, 201; method=:euler_maruyama, dev)

# # # # plot_total_energy_over_time(all_groundtruth_batch, all_u_les_batch, all_u_model_batch, all_u_deterministic_batch, dt_LES, "total_energy_Heun_50.png", batch_size)
# # # # probability_density_energy(all_u_model_batch, all_groundtruth_batch, all_u_deterministic_batch, 0.05, batch_size, 51; method=:euler_maruyama)
# # # # probability_density_energy(all_u_model_batch, all_groundtruth_batch, all_u_deterministic_batch, 0.1, batch_size, 101; method=:euler_maruyama)
# # # # probability_density_energy(all_u_model_batch, all_groundtruth_batch, all_u_deterministic_batch, 0.15, batch_size, 151; method=:euler_maruyama)
# # # # probability_density_energy(all_u_model_batch, all_groundtruth_batch, all_u_deterministic_batch, 0.2, batch_size, 201; method=:euler_maruyama)

# # # # snapshot_comparison(51, dt_LES, all_groundtruth_batch, all_u_les_batch, all_u_model_batch, all_u_deterministic_batch, N_les, Re, batch_size, "figures"; dev)
# # # # snapshot_comparison(101, dt_LES, all_groundtruth_batch, all_u_les_batch, all_u_model_batch, all_u_deterministic_batch, N_les, Re, batch_size, "figures"; dev)
# # # # snapshot_comparison(151, dt_LES, all_groundtruth_batch, all_u_les_batch, all_u_model_batch, all_u_deterministic_batch, N_les, Re, batch_size, "figures"; dev)
# # # # snapshot_comparison(201, dt_LES, all_groundtruth_batch, all_u_les_batch, all_u_model_batch, all_u_deterministic_batch, N_les, Re, batch_size, "figures"; dev)

























# # # #### Is the data correctly generated?
# # # function compute_energy(test_labels_state)
# # #     nt = size(test_labels_state, 4)
# # #     energy = zeros(nt)  
# # #     for i in 1:nt
# # #         u = test_labels_state[:,:,1,i]
# # #         v = test_labels_state[:,:,2,i] 
# # #         energy[i] = 0.5 * sum(u.^2 .+ v.^2) 
# # #     end
# # #     return energy
# # # end

# # # function plot_energy_over_time(test_labels_state, dt_LES, save_path="energy_evolution.png")
# # #     energy = compute_energy(test_labels_state)
# # #     time = (0:size(test_labels_state, 4)-1) * dt_LES 
# # #     plot(time, energy, label="Ground Truth Energy", xlabel="Time", ylabel="Total Energy", title="Energy Evolution")
# # #     t_critical = 0.15
# # #     idx_critical = findfirst(x -> x > t_critical, time)
# # #     if isnothing(idx_critical)
# # #         println("Warning: No time point found after t=0.15")
# # #         idx_critical = length(time)
# # #     end
# # #     idx_before = max(1, idx_critical - 1) 
# # #     idx_after = min(length(time), idx_critical + 1) 
# # #     scatter!([time[idx_before], time[idx_critical], time[idx_after]], 
# # #              [energy[idx_before], energy[idx_critical], energy[idx_after]], 
# # #              label="Before, At, After t=0.15", color=:red, markersize=5)
# # #     savefig(save_path) 
# # #     println("Plot saved to: $save_path")
# # # end

# # # plot_energy_over_time(train_data.state[:,:,:,:,1], dt_LES, "energy_plot_1.png")
# # # plot_energy_over_time(train_data.state[:,:,:,:,2], dt_LES, "energy_plot_2.png")
# # # plot_energy_over_time(train_data.state[:,:,:,:,3], dt_LES, "energy_plot_3.png")
# # # plot_energy_over_time(train_data.state[:,:,:,:,4], dt_LES, "energy_plot_4.png")
# # # plot_energy_over_time(train_data.state[:,:,:,:,5], dt_LES, "energy_plot_5.png")
# # # plot_energy_over_time(train_data.state[:,:,:,:,6], dt_LES, "energy_plot_6.png")
# # # plot_energy_over_time(train_data.state[:,:,:,:,7], dt_LES, "energy_plot_7.png")
# # # plot_energy_over_time(train_data.state[:,:,:,:,8], dt_LES, "energy_plot_8.png")
# # # plot_energy_over_time(train_data.state[:,:,:,:,9], dt_LES, "energy_plot_9.png")
# # # plot_energy_over_time(train_data.state[:,:,:,:,10], dt_LES, "energy_plot_10.png")
# # # plot_energy_over_time(train_data.state[:,:,:,:,11], dt_LES, "energy_plot_11.png")
# # # plot_energy_over_time(train_data.state[:,:,:,:,12], dt_LES, "energy_plot_12.png")
# # # plot_energy_over_time(train_data.state[:,:,:,:,13], dt_LES, "energy_plot_13.png")
# # # plot_energy_over_time(train_data.state[:,:,:,:,14], dt_LES, "energy_plot_14.png")
# # # plot_energy_over_time(train_data.state[:,:,:,:,15], dt_LES, "energy_plot_15.png")
# # # plot_energy_over_time(train_data.state[:,:,:,:,16], dt_LES, "energy_plot_16.png")
# # # plot_energy_over_time(train_data.state[:,:,:,:,17], dt_LES, "energy_plot_17.png")
# # # plot_energy_over_time(train_data.state[:,:,:,:,18], dt_LES, "energy_plot_18.png")
# # # plot_energy_over_time(train_data.state[:,:,:,:,19], dt_LES, "energy_plot_19.png")
# # # plot_energy_over_time(train_data.state[:,:,:,:,20], dt_LES, "energy_plot_20.png")
# # # plot_energy_over_time(train_data.state[:,:,:,:,21], dt_LES, "energy_plot_21.png")
# # # plot_energy_over_time(train_data.state[:,:,:,:,22], dt_LES, "energy_plot_22.png")
# # # plot_energy_over_time(train_data.state[:,:,:,:,23], dt_LES, "energy_plot_23.png")
# # # plot_energy_over_time(train_data.state[:,:,:,:,24], dt_LES, "energy_plot_24.png")
# # # plot_energy_over_time(val_data.state[:,:,:,:,1], dt_LES, "energy_plot_25.png")
# # # plot_energy_over_time(val_data.state[:,:,:,:,2], dt_LES, "energy_plot_26.png")
# # # plot_energy_over_time(val_data.state[:,:,:,:,3], dt_LES, "energy_plot_27.png")
# # # plot_energy_over_time(val_data.state[:,:,:,:,4], dt_LES, "energy_plot_28.png")
# # # plot_energy_over_time(test_data.state[:,:,:,:,1], dt_LES, "energy_plot_29.png")
# # # plot_energy_over_time(test_data.state[:,:,:,:,2], dt_LES, "energy_plot_30.png")
# # # plot_energy_over_time(test_data.state[:,:,:,:,3], dt_LES, "energy_plot_31.png")
# # # plot_energy_over_time(test_data.state[:,:,:,:,4], dt_LES, "energy_plot_32.png")

# ####### Example SI #######
# time_steps = [0.0, 0.25, 0.5, 0.75, 1.0]

# image1 = train_data.initial[:,:,:,1,1] |> dev;
# image2 = train_data.target[:,:,:,10,1] |> dev;
# slice_plots = []
        
# for slice_idx in 1:2
#     row_plots = []
#     for λ in time_steps
#         W = sqrt(λ) .* randn(size(image1,1), size(image2,2), 2) |> dev
#         interpolated = stochastic_interpolant(image1[:, :, slice_idx], image2[:, :, slice_idx], W[:, :, slice_idx], λ, ϵ)
#         push!(row_plots, heatmap(Array(interpolated), color=:viridis, title="λ = $λ, slice = $slice_idx", aspect_ratio=:equal,  clims=(-30,30)))
#     end
#     println("check 3")
#     push!(slice_plots, plot(row_plots..., layout=(1, length(time_steps)), size=(2000, 400)))
# end
# println("check 4")
# final_plot = plot(slice_plots..., layout=(2, 1), size=(2000, 800))
# savefig(final_plot, "stochastic_interpolant.png")


# ####### Example SI #######
# using Random, Plots

# num_samples = 5;
# num_s = 100;
# filename="I_s_transition.png"
# s_values = range(0, 1, length=num_s)
# x_0 = 0
# x_1_samples = rand(num_samples)
# I_s_values = [ (1 - s) * x_0 + s^2 * x_1 + (1 - s) * sqrt(s) * randn() for s in s_values, x_1 in x_1_samples ]
# plt = plot(title="Different Trajectories from Fixed x₀ to Multiple x₁", xlabel="s", ylabel="I_s", legend=false, grid=true, titlefontsize=14, guidefontsize=12, tickfontsize=12)
# for i in 1:num_samples
#     plot!(plt, s_values, I_s_values[:, i], lw=2, alpha=0.7)
# end
# savefig(plt, "I_s_transition.png")

# num_paths = 5;
# s_values = range(0, 1, length=num_s)
# x_0 = 0
# x_1 = rand() 
# I_s_values = [ (1 - s) * x_0 + s^2 * x_1 + (1 - s) * sqrt(s) * randn() for s in s_values, _ in 1:num_paths ]
# plt = plot(title="Multiple Stochastic Paths for Fixed (x₀, x₁)", xlabel="s", ylabel="I_s", legend=false, grid=true, titlefontsize=14, guidefontsize=12, tickfontsize=12)
# for i in 1:num_paths
#     plot!(plt, s_values, I_s_values[:, i], lw=2, alpha=0.7)
# end
# savefig(plt, "I_s_transition_ind.png")


# ######## FIGURE INTRODUCTION ########
# trajectory_to_evaluate = 2; 

# create_right_hand_side(setup, psolver) = function right_hand_side(u, p, t)
#     u = pad_circular(u, 1; dims = 1:2)
#     F = INS.momentum(u, nothing, t, setup)
#     F = F[2:end-1, 2:end-1, :]
#     F = pad_circular(F, 1; dims = 1:2)
#     PF = INS.project(F, setup; psolver)
#     PF[2:end-1, 2:end-1, :]
# end

# backend = CUDABackend()
# x_les = LinRange(0.0, 1.0, N_les + 1), LinRange(0.0, 1.0, N_les + 1)
# setup_les = INS.Setup(; x=x_les, Re=Re, backend)
# psolver_les = INS.psolver_spectral(setup_les)
# f_les = create_right_hand_side(setup_les, psolver_les)

# global u = train_data.state[:,:,:,1,trajectory_to_evaluate]
# global t = 0
# v = [];
# nt = 200;
# for i = 1:nt+1
#     if i > 1
#         global u, t
#         t += dt_LES
#         u = step_rk4(u, dt_LES, f_les)
#     end
#     push!(v, Array(u))
# end
# v_generated = CuArray(cat(v..., dims=4))
# v_true = CuArray(train_data.state[:,:,:,:,trajectory_to_evaluate])

# ω_generated_0 = Array(INS.vorticity(pad_circular(v_generated[:,:,:,1], 1; dims=1:2), setup_les)[2:end-1, 2:end-1]) 
# ω_true_0 = Array(INS.vorticity(pad_circular(v_true[:,:,:,1], 1; dims=1:2), setup_les)[2:end-1, 2:end-1])

# ω_generated_100 = Array(INS.vorticity(pad_circular(v_generated[:,:,:,101], 1; dims=1:2), setup_les)[2:end-1, 2:end-1]) 
# ω_true_100 = Array(INS.vorticity(pad_circular(v_true[:,:,:,101], 1; dims=1:2), setup_les)[2:end-1, 2:end-1])

# ω_generated_200 = Array(INS.vorticity(pad_circular(v_generated[:,:,:,201], 1; dims=1:2), setup_les)[2:end-1, 2:end-1]) 
# ω_true_200 = Array(INS.vorticity(pad_circular(v_true[:,:,:,201], 1; dims=1:2), setup_les)[2:end-1, 2:end-1])

# p1 = heatmap(ω_generated_0, title="Generated ω at t=0", color=:viridis, xlabel="x", ylabel="y")
# p2 = heatmap(ω_generated_100, title="Generated ω at t=100", color=:viridis, xlabel="x", ylabel="y")
# p3 = heatmap(ω_generated_200, title="Generated ω at t=200", color=:viridis, xlabel="x", ylabel="y")

# p4 = heatmap(ω_true_0, title="True ω at t=0", color=:viridis, xlabel="x", ylabel="y")
# p5 = heatmap(ω_true_100, title="True ω at t=100", color=:viridis, xlabel="x", ylabel="y")
# p6 = heatmap(ω_true_200, title="True ω at t=200", color=:viridis, xlabel="x", ylabel="y")

# fig = plot(p1, p2, p3, p4, p5, p6, layout=(2, 3), size=(1200, 800), aspect_ratio=:equal)
# savefig(fig, "vorticity_comparison.png")

# # ####### TEST MAGNITUDE OF SIZES #######
# # trajectory_to_evaluate = 2; 
# # initial_test_images, test_images, test_labels_closure, test_labels_state = test_data.initial, test_data.target, test_data.closure, test_data.state |> dev;
# # initial_test_images, test_images, test_labels_closure, test_labels_state = test_data.initial[:,:,:,:,trajectory_to_evaluate], test_data.target[:,:,:,:,trajectory_to_evaluate], test_data.closure[:,:,:,:,trajectory_to_evaluate], test_data.state[:,:,:,:,trajectory_to_evaluate] |> dev; 

# # create_right_hand_side(setup, psolver) = function right_hand_side(u, p, t)
# #     u = pad_circular(u, 1; dims = 1:2)
# #     F = INS.momentum(u, nothing, t, setup)
# #     F = F[2:end-1, 2:end-1, :]
# #     F = pad_circular(F, 1; dims = 1:2)
# #     PF = INS.project(F, setup; psolver)
# #     PF[2:end-1, 2:end-1, :]
# # end

# # create_right_hand_side_non_proj(setup) = function right_hand_side(u, p, t)
# #     u = pad_circular(u, 1; dims = 1:2)
# #     F = INS.momentum(u, nothing, t, setup)
# #     F = F[2:end-1, 2:end-1, :]
# # end

# # backend = CUDABackend();
# # x_les = LinRange(0.0, 1.0, N_les + 1), LinRange(0.0, 1.0, N_les + 1);
# # setup_les = INS.Setup(; x=x_les, Re=Re, backend);
# # psolver_les = INS.psolver_spectral(setup_les);
# # f_les = create_right_hand_side(setup_les, psolver_les); 
# # f_les_non_proj = create_right_hand_side_non_proj(setup_les);


# # u_les = test_labels_state[:,:,:,1]; 
# # closure = test_images[:,:,:,1];

# # closure_pad = pad_circular(closure, 1; dims = 1:2) |> dev;
# # proj_pad_done = INS.project(closure_pad, setup_les; psolver=psolver_les) |> dev;
# # proj_closure = proj_pad_done[2:end-1, 2:end-1, :] |> dev;


# # F_u = f_les_non_proj(u_les, nothing, 0.0); 
# # F_u_proj = f_les(u_les, nothing, 0.0); 

# # mag_u_les = norm(u_les[:])
# # mag_closure = norm(closure[:])
# # mag_closure_proj = norm(proj_closure[:])
# # mag_F_u = norm(F_u[:])
# # mag_F_u_proj = norm(F_u_proj[:])

# # # Print results
# # println("Magnitude of u_les: ", mag_u_les)
# # println("Magnitude of closure: ", mag_closure)
# # println("Magnitude of projected closure: ", mag_closure_proj)
# # println("Magnitude of F_u: ", mag_F_u)
# # println("Magnitude of F_u_proj: ", mag_F_u_proj)

# ###### Create loss plot with bigger fontsize ######
# model_name = "big_closure_to_closure_explicit"; 
# state = load_training_state("trained_models", "big_closure_to_closure_explicit")
# drift_losses = state[:drift_losses]
# test_drift_losses = state[:test_drift_losses]

# p1 = plot(1:length(drift_losses), drift_losses, label="Training Loss", xlabel="Epoch", ylabel="Loss", title="Training and Validation Loss", yscale=:log10, guidefontsize=14, tickfontsize=12, legendfontsize=12)
# plot!(p1, 1:length(test_drift_losses), test_drift_losses, label="Validation Loss")
# savefig(p1, "figures/final_loss_plot_$(model_name).png")

# quality_deterministic_model(test_data, velocity_cnn_det, ps_deterministic, _st_deterministic, setup_les, psolver_les, N_les; dev)
# time_steps_dependency(test_data, velocity_cnn, ps_drift, _st_drift, ϵ, setup_les, psolver_les, N_les; dev, method=:heuns_method)