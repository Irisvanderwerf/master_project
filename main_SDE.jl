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
nt = 40; # Number of time steps (number of training and test samples)

# Set paths to save/load data and choose if you want to generate new data. 
v_train_path = "datasets/v_train_data_FVM_small.bson";
c_train_path = "datasets/c_train_data_FVM_small.bson";
v_test_path = "datasets/v_test_data_FVM_small.bson";
c_test_path = "datasets/c_test_data_FVM_small.bson";

v_train_standardized_path = "datasets/v_train_stand_data_FVM_small.bson";
c_train_standardized_path = "datasets/c_train_stand_data_FVM_small.bson";
v_test_standardized_path = "datasets/v_test_stand_data_FVM_small.bson";
c_test_standardized_path = "datasets/c_test_stand_data_FVM_small.bson";

state_means_path = "datasets/state_means_small.pkl"
state_std_path = "datasets/state_std_small.pkl"
closure_means_path = "datasets/closure_means_small.pkl"
closure_std_path = "datasets/closure_std_small .pkl"

generate_new_data = false;
generate_new_stand_data = false;

v_train, c_train, v_test, c_test = generate_or_load_data(N_dns, N_les, Re, v_train_path, c_train_path, v_test_path, c_test_path, generate_new_data, nt, dt, num_initial_conditions, num_train_conditions; dev);
v_train_standardized, c_train_standardized, v_test_standardized, c_test_standardized, state_means, state_std, closure_means, closure_std = generate_or_load_standardized_data(v_train_standardized_path, c_train_standardized_path, v_test_standardized_path, c_test_standardized_path, generate_new_stand_data, v_train, c_train, v_test, c_test, state_means_path, state_std_path, closure_means_path, closure_std_path);0

####### INITIALIZE/LOAD NETWORK ####### - two seperate models for the denoiser and drift term in complexity. 
velocity_cnn = build_full_unet(16,[32,64,128,256],128; dev)
model_name = "SDE_closure_time_stepping_cond_on_current_state";
load_path = nothing; 
is_gaussian = false; 

ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser = initialize_or_load_model(model_name, velocity_cnn, load_path; dev, method=:SDE, is_gaussian)

###### MAKE TRAINING SETS READY ######
initial, target, target_label = create_training_sets(c_train_standardized, v_train_standardized); 
initial_test, target_test, target_label_test = create_training_sets(c_test_standardized, v_test_standardized);

####### TRAINING #######
batch_size = 32;
num_epochs = 10;
train!(initial, target, target_label, batch_size, num_epochs, ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser, velocity_cnn, target_test, initial_test, target_label_test, "trained_models", model_name; is_gaussian, method=:SDE, dev)
println("The network is trained")

# ###### EVALUATION ######
# load_path = "trained_models/$model_name.bson"
# ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser = initialize_or_load_model(model_name, velocity_cnn, load_path; dev, method=:SDE, is_gaussian); 
# _st_drift = Lux.testmode(st_drift) |> dev;
# _st_denoiser = Lux.testmode(st_denoiser) |> dev;

# num_steps = 100; 
# trajectory_to_evaluate = rand(1:(num_initial_conditions-num_train_conditions));
# initial_test = initial_test[:,:,:,:,trajectory_to_evaluate];
# target_label_test = target_label_test[:,:,:,:,trajectory_to_evaluate];

# predicted_closure = generate_closure(velocity_cnn, ps_drift, _st_drift, ps_denoiser, _st_denoiser, target_label_test, initial_test, num_steps, is_gaussian, dev; method=:SDE)
# closure = inverse_standardize_set_per_channel(predicted_closure, closure_means, closure_std);

# # Compute the metrices. 
# compute_metrics_average(closure, target_test[:,:,:,:,trajectory_to_evaluate]; epsilon=1e-8, dev)

# ###### INFERENCE ######
# batch_size = 1; 
# nt = 50; 
# num_steps = 100;

# v_test = v_test[:,:,:,:,trajectory_to_evaluate];
# inference_deterministic(dt, nt, batch_size, v_test, velocity_cnn, ps_drift, _st_drift, closure_means, closure_std, state_means, state_std, N_les, Re, dev);