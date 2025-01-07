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

data_v, data_c = generate_or_load_data(N_dns, N_les, Re, output_dir, generate_new_data, nt, dt, num_trajectories; dev);
stand_data_v, stand_data_c, stats = generate_or_load_stand_data(data_v, data_c, standardized_dir, generate_new_stand_data) |> dev;

initial_sample, target_sample, target_label_closure, target_label_state = create_training_sets(stand_data_c, stand_data_v, N_les);
splits = split_trajectories(initial_sample, target_sample, target_label_closure, target_label_state); 

train_data = splits.train
val_data = splits.val
test_data = splits.test

####### TRAINING ####### 
velocity_cnn = build_full_unet(16,[32,64,128,256],128; dev)
model_name = "closure_to_closure";
load_path = nothing;
ps_drift, st_drift, opt_drift = initialize_or_load_model(model_name, velocity_cnn, load_path; dev);

num_training_time_steps = 500;
eval_frequency = 5; 
val_subset_size = 100; 
batch_size = 4;
num_epochs = 100;
train!(train_data, val_data, batch_size, num_epochs, ps_drift, st_drift, opt_drift, velocity_cnn, "trained_models", model_name, num_training_time_steps, eval_frequency, val_subset_size; dev);

####### INFERENCE #######


















# # ###### EVALUATION ######
# # load_path = "trained_models/$model_name.bson"
# # ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser = initialize_or_load_model(model_name, velocity_cnn, load_path; dev, method=:SDE, is_gaussian); 
# # _st_drift = Lux.testmode(st_drift) |> dev;
# # _st_denoiser = Lux.testmode(st_denoiser) |> dev;

# # num_steps = 100; 
# # trajectory_to_evaluate = rand(1:(num_initial_conditions-num_train_conditions));
# # initial_test = initial_test[:,:,:,:,trajectory_to_evaluate];
# # target_label_test = target_label_test[:,:,:,:,trajectory_to_evaluate];

# # predicted_closure = generate_closure(velocity_cnn, ps_drift, _st_drift, ps_denoiser, _st_denoiser, target_label_test, initial_test, num_steps, is_gaussian, dev; method=:SDE)
# # closure = inverse_standardize_set_per_channel(predicted_closure, closure_means, closure_std);

# # # Compute the metrices. 
# # compute_metrics_average(closure, target_test[:,:,:,:,trajectory_to_evaluate]; epsilon=1e-8, dev)

# # ###### INFERENCE ######
# # batch_size = 1; 
# # nt = 50; 
# # num_steps = 100;

# # v_test = v_test[:,:,:,:,trajectory_to_evaluate];
# # inference_deterministic(dt, nt, batch_size, v_test, velocity_cnn, ps_drift, _st_drift, closure_means, closure_std, state_means, state_std, N_les, Re, dev);