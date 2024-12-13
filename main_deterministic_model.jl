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

####### DATA GENERATION: Training + Test set #######
# Parameters. 
Re = 1.0f3;
num_initial_conditions = 32; # num_train_conditions < num_initial_conditions
num_train_conditions = 28; 
N_les = 64; # [N_{les} < N_{dns}]
N_dns = 256; 
dt = 1.0f-4; # Time step Δt
nt = 200; # Number of time steps (number of training and test samples)

# Set paths to save/load data and choose if you want to generate new data. 
v_train_path = "datasets/v_train_data_FVM.bson";
c_train_path = "datasets/c_train_data_FVM.bson";
v_test_path = "datasets/v_test_data_FVM.bson";
c_test_path = "datasets/c_test_data_FVM.bson";

v_train_standardized_path = "datasets/v_train_stand_data_FVM.bson";
c_train_standardized_path = "datasets/c_train_stand_data_FVM.bson";
v_test_standardized_path = "datasets/v_test_stand_data_FVM.bson";
c_test_standardized_path = "datasets/c_test_stand_data_FVM.bson";

state_means_path = "datasets/state_means.pkl"
state_std_path = "datasets/state_std.pkl"
closure_means_path = "datasets/closure_means.pkl"
closure_std_path = "datasets/closure_std.pkl"

generate_new_data = false;
generate_new_stand_data = false;

v_train, c_train, v_test, c_test = generate_or_load_data(N_dns, N_les, Re, v_train_path, c_train_path, v_test_path, c_test_path, generate_new_data, nt, dt, num_initial_conditions, num_train_conditions; dev);
v_train_standardized, c_train_standardized, v_test_standardized, c_test_standardized, state_means, state_std, closure_means, closure_std = generate_or_load_standardized_data(v_train_standardized_path, c_train_standardized_path, v_test_standardized_path, c_test_standardized_path, generate_new_stand_data, v_train, c_train, v_test, c_test, state_means_path, state_std_path, closure_means_path, closure_std_path);0

####### INITIALIZE/LOAD NETWORK #######
velocity_cnn = build_full_unet(16,[32,64,128],64; dev) 
model_name = "Deterministic_medium_bigger";
load_path = nothing; 
is_gaussian = true; 

ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser = initialize_or_load_model(model_name, velocity_cnn, load_path; dev, method=:ODE, is_gaussian)
println("The network is ready")

####### TRAINING #######
batch_size = 32;
num_epochs = 500;
train_deterministic!(velocity_cnn, ps_drift, st_drift, opt_drift, num_epochs, batch_size, c_train_standardized, v_train_standardized, model_name, "trained_models", v_test_standardized, c_test_standardized; dev) 
println("The network is trained")

###### EVALUATION ######
# load_path = "trained_models/$model_name.bson"
# ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser = initialize_or_load_model(model_name, velocity_cnn, load_path; dev, method=:ODE, is_gaussian) 
_st_drift = Lux.testmode(st_drift) |> dev;

trajectory_to_evaluate = rand(1:(num_initial_conditions-num_train_conditions));

v_test_standardized = v_test_standardized[:,:,:,:,trajectory_to_evaluate] |> dev; 
t_sample = Float32.(fill(0, 1, 1, 1, size(v_test_standardized, 4))) |> dev;
predicted_closure, _ = Lux.apply(velocity_cnn, (v_test_standardized, t_sample, v_test_standardized), ps_drift, _st_drift) |> dev;
closure = inverse_standardize_set_per_channel(predicted_closure, closure_means, closure_std);

compute_metrics_average(closure, c_test[:,:,:,:,trajectory_to_evaluate]; epsilon=1e-8, dev); 

###### INFERENCE ######
batch_size = 1; 
nt = 3000; 

v_test = v_test[:,:,:,:,trajectory_to_evaluate] |> dev; 
inference_deterministic(dt, nt, batch_size, v_test, velocity_cnn, ps_drift, _st_drift, closure_means, closure_std, state_means, state_std, N_les, Re; dev);