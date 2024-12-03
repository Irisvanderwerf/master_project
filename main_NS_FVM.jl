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
Re = 1.0f4;
num_initial_conditions = 32; # num_train_conditions < num_initial_conditions
num_train_conditions = 28; 
N_les = 64; # [N_{les} < N_{dns}]
N_dns = 512; 
dt = 1.0f-4; # Time step Δt
nt = 1000; # 16000; # Number of time steps (number of training and test samples)

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

generate_new_data = true;
generate_new_stand_data = true;

v_train, c_train, v_test, c_test = generate_or_load_data(N_dns, N_les, Re, v_train_path, c_train_path, v_test_path, c_test_path, generate_new_data, nt, dt, num_initial_conditions, num_train_conditions; dev);
v_train_standardized, c_train_standardized, v_test_standardized, c_test_standardized, state_means, state_std, closure_means, closure_std = generate_or_load_standardized_data(v_train_standardized_path, c_train_standardized_path, v_test_standardized_path, c_test_standardized_path, generate_new_stand_data, v_train, c_train, v_test, c_test, state_means_path, state_std_path, closure_means_path, closure_std_path);
println("Dataset ready for use, which is of size: ", size(v_train))

# ####### INITIALIZE/LOAD NETWORK #######
velocity_cnn = build_full_unet(16,[32,64,128],8; dev); # u-net with ConvNextBlocks. 
is_gaussian = true; # If the initial condition is a Gaussian or not.
model_name = "ODE_model_state[i]_to_closure[i]";
load_path = nothing; "trained_models/$model_name.bson"; # Set to path of your model to load your model or to nothing if you initialize the model.

ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser = initialize_or_load_model(model_name, velocity_cnn, load_path; dev, method=:ODE, is_gaussian); # Set method to ODE or SDE. 
println("The network is ready")

####### TRAINING #######
batch_size = 32;
num_epochs = 500;

train!(velocity_cnn, ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser, num_epochs, batch_size, v_train_standardized, c_train_standardized, v_train_standardized, model_name, "trained_models", v_test, c_test; dev, method=:ODE, is_gaussian)
println("The network is trained")

####### EVALUATION OF THE NETWORK #######
_st_drift = Lux.testmode(st_drift) |> dev;
if !is_gaussian
    _st_denoiser = Lux.testmode(st_denoiser) |> dev;
else
    _st_denoiser = nothing;
end
println(" Set state parameters into test mode")

# Determine the computed predicted closure for the test set. 
num_steps = 100; 
predicted_closure = generate_closure(velocity_cnn, ps_drift, _st_drift, ps_denoiser, _st_denoiser, v_test_standardized, v_test_standardized, num_steps, is_gaussian, dev; time_method=:rk4, method=:ODE)
println(" The closure is predicted for the test set which is of size: ", size(predicted_closure))

# Compute the metrices. 
compute_metrics_average(predicted_closure, c_test_standardized; epsilon=1e-8, dev)

# ####### INFERENCE #######
batch_size = 1; # how many samples do you want to evolve with the same initial condition. 
nt = 100; 
num_steps = 100;
inference(dt, nt, num_steps, batch_size, v_test_standardized, v_test, velocity_cnn, ps_drift, _st_drift, ps_denoiser, _st_denoiser, is_gaussian, closure_means, closure_std, state_means, state_std, N_les, Re, dev; time_method=:euler, method=:ODE)
