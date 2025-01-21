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

dev = gpu_device()
if dev==gpu_device()
    gr()
    ENV[" GKSwstype"] = "100"
    CUDA.allowscalar(false)
end

####### DATA GENERATION ####### 
output_dir = "datasets/trajectories";
standardized_dir = "datasets/standardized_trajectories";
generate_new_data = false;
generate_new_stand_data = false;
Re = 1.0f4;
num_trajectories = 32;
N_les = 64; 
N_dns = 2048; 
dt = 1.0f-4;
nt = 100000; 

ϵ = 1;

data_v, data_c = generate_or_load_data(N_dns, N_les, Re, output_dir, generate_new_data, nt, dt, num_trajectories; dev);
stand_data_v, stand_data_c, stats = generate_or_load_stand_data(data_v, data_c, standardized_dir, generate_new_stand_data) |> dev;

initial_sample, target_sample, target_label_closure, target_label_state = create_training_sets(stand_data_c, stand_data_v, N_les);
splits = split_trajectories(initial_sample, target_sample, target_label_closure, target_label_state); 

train_data = splits.train
val_data = splits.val
test_data = splits.test

####### INITIALIZE/LOAD MODEL ####### 
velocity_cnn = build_full_unet(16,[32,64,128,256],128; dev)
model_name = "closure_to_closure_big";
load_path = "trained_models/$model_name.bson";
ps_drift, st_drift, opt_drift = initialize_or_load_model(model_name, velocity_cnn, load_path; dev) |> dev;

####### TRAINING #######
num_training_time_steps = 3000; 
batch_size = 32;
num_epochs = 100;
train!(train_data, val_data, batch_size, num_epochs, ps_drift, st_drift, opt_drift, velocity_cnn, "trained_models", model_name, num_training_time_steps, load_path; dev);

####### EVALUATION SDE #######
_st_drift = Lux.testmode(st_drift) |> dev;

plot_closure_prediction(test_data, 75, 1, 10, velocity_cnn, ps_drift, _st_drift, ϵ; dev, method=:euler_maruyama)
plot_closure_prediction(test_data, 75, 1, 100, velocity_cnn, ps_drift, _st_drift, ϵ; dev, method=:heuns_method)

time_steps_dependency(test_data, velocity_cnn, ps_drift, _st_drift, ϵ, model_name; dev, method=:euler_maruyama)
time_steps_dependency(test_data, velocity_cnn, ps_drift, _st_drift, ϵ, model_name; dev, method=:heuns_method)

####### INFERENCE #######
_st_drift = Lux.testmode(st_drift) |> dev;
nt = 100;
num_steps = 20;
dt_LES = 10 * dt; 
trajectory_to_evaluate = 4;
batch_size = 10;

all_groundtruth_batch, all_u_les_batch, all_u_model_batch = batched_inference(N_les, batch_size, dt_LES, num_steps, nt, test_data, stats, velocity_cnn, ps_drift, _st_drift, Re, trajectory_to_evaluate, ϵ, dev; method=:euler_maruyama) |> dev; 

####### EVALUATION OF INFERENCE: GENERATION OF PLOTS #######
println(" Start plotting evaluation results ")
error_comparison_LES_model(all_groundtruth_batch, all_u_les_batch, all_u_model_batch, dt_LES, "figures/$model_name/error.png")
plot_total_energy_over_time(all_groundtruth_batch, all_u_les_batch, all_u_model_batch, dt_LES, "figures/$model_name/total_energy.png", batch_size)
animation_LES_model_truth(all_u_les_batch, all_u_model_batch, all_groundtruth_batch, nt, dt, N_les, Re, "figures/$model_name/animation.mp4")
snapshot_comparison(20, dt_LES, all_groundtruth_batch, all_u_les_batch, all_u_model_batch, N_les, Re, batch_size, nt, "figures/$model_name"; dev)
snapshot_comparison(40, dt_LES, all_groundtruth_batch, all_u_les_batch, all_u_model_batch, N_les, Re, batch_size, nt, "figures/$model_name"; dev)
snapshot_comparison(60, dt_LES, all_groundtruth_batch, all_u_les_batch, all_u_model_batch, N_les, Re, batch_size, nt, "figures/$model_name"; dev)
snapshot_comparison(80, dt_LES, all_groundtruth_batch, all_u_les_batch, all_u_model_batch, N_les, Re, batch_size, nt, "figures/$model_name"; dev)
snapshot_comparison(100, dt_LES, all_groundtruth_batch, all_u_les_batch, all_u_model_batch, N_les, Re, batch_size, nt, "figures/$model_name"; dev)






















# ####### CHECK ENERGY SPECTRUM OF THE INITIALIZATION #######
# N_dns = 2048;
# Re=1.0f4;
# backend = CUDABackend()
# x_dns = LinRange(0.0, 1.0, N_dns + 1), LinRange(0.0, 1.0, N_dns + 1)
# setup_dns = INS.Setup(; x=x_dns, Re=Re, backend)
# psolver_dns = INS.psolver_spectral(setup_dns)
# f_dns = create_right_hand_side(setup_dns, psolver_dns)

# global u = INS.random_field(setup_dns, 0.0)
# unew = u; 
# println(" size of u: ", size(u))
# nburn = 5000
# dt=1.0f-4;
# for i = 1:nburn
#     global unew = step_rk4(unew, dt, f_dns)
# end
# u = u[2:end-1, 2:end-1, :]
# unew = unew[2:end-1, 2:end-1,:]
# println(" the size of u: ", size(u), " and the size of updated u: ", size(unew))
# energy, K_bins = compute_energy_spectra(Array(u))
# energy_new, _ = compute_energy_spectra(Array(unew))
# energy_spectrum_plot = Plots.plot(K_bins, energy[:, 1], label="Initial Energy spectrum", xaxis=:log, yaxis=:log, xlabel="log κ", ylabel="log E_{κ}(κ)", title="Energy Spectrums")
# energy_spectrum_plot_new = Plots.plot!(energy_spectrum_plot, K_bins, energy_new[:,1], label="Energy spectrum after n_burn")
# savefig(energy_spectrum_plot, "figures/initial_energy_spectrum.png")
