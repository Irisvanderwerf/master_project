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
Re = 2.0f4;
num_initial_conditions = 32;
num_train_conditions = 28;
N_les = 64; # [N_{les} < N_{dns}]
N_dns = 512; 
dt = 1.0f-4; # Time step Δt
nt = 128000; # Number of time steps (number of training and test samples)

# Set paths to save/load data and choose if you want to generate new data. 
v_train_path = "datasets/v_train_data_FVM.bson";
c_train_path = "datasets/c_train_data_FVM.bson";
v_test_path = "datasets/v_test_data_FVM.bson";
c_test_path = "datasets/c_test_data_FVM.bson";

v_train_standardized_path = "datasets/v_train_stand_data_FVM.bson";
c_train_standardized_path = "datasets/c_train_stand_data_FVM.bson";
v_test_standardized_path = "datasets/v_test_stand_data_FVM.bson";
c_test_standardized_path = "datasets/c_test_stand_data_FVM.bson";

generate_new_data = true;

v_train, c_train, v_test, c_test = generate_or_load_data(N_dns, N_les, Re, v_train_path, c_train_path, v_test_path, c_test_path, generate_new_data, nt, dt, num_initial_conditions, num_train_conditions; dev);
v_train_standardized, c_train_standardized, v_test_standardized, c_test_standardized, state_means, state_std, closure_means, closure_std = generate_or_load_standardized_data(v_train_standardized_path, c_train_standardized_path, v_test_standardized_path, c_test_standardized_path, generate_new_data, v_train, c_train, v_test, c_test);
println(" The training set has size, ", size(v_train, 4), " and the test set has size, ", size(v_test, 4))

####### INITIALIZE NETWORK #######


# # Compute mean and standard deviation for `v_train` and standardize the set. 
# state_means, state_std = compute_mean_std(v_train);
# v_train_standardized = standardize_training_set_per_channel(v_train, state_means, state_std);

# # Compute mean and standard deviation for `c_train` and standardize the set.
# closure_means, closure_std = compute_mean_std(c_train);
# c_train_standardized = standardize_training_set_per_channel(c_train, closure_means, closure_std);

# # Define the network - u-net with ConvNextBlocks 
# velocity_cnn = build_full_unet(16,[32,64,128],8; dev);

# # Is the initial data gaussian distributed?
# is_gaussian = true; 
# # Name of potential new model
# model_name = "gaussian_model_FVM";
# # Path of potential loaded model. 
# load_path = "trained_models/gaussian_model_FVM.bson";  # Set to `nothing` to initialize instead

# # Load/Initialize parameters of the network - ODE
# ps_drift, st_drift, opt_drift = initialize_or_load_model(model_name, velocity_cnn, load_path; dev);

# # Define the batch_size, num_batches, num_epochs
# batch_size = 32;
# num_samples = size(c_train,4);
# num_batches = ceil(Int, num_samples / batch_size);

# num_epochs = 100;

# # Train ODE 
# train!(velocity_cnn, ps_drift, st_drift, opt_drift, num_epochs, batch_size, c_train_standardized, v_train, num_batches, dev, model_name, "trained_models");
# # # Train SDE
# # # train!(velocity_cnn, ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser, num_epochs, batch_size, v_train, c_train, v_train, num_batches, dev, is_gaussian, model_name, "trained_models");

# #### USE TRAINED MODEL #####;
# # # Set to test mode
# _st_drift = Lux.testmode(st_drift) |> dev;
# # _st_denoiser = Lux.testmode(st_denoiser) |> dev; # (Add when using the SDE)

# # Compute the error between the predicted closure and the real closure of a random set of the training data.
# num_steps = 100; # For the ODE evaluation. 
# max_difference = -Inf;
# min_difference = Inf;

# num_samples = min(1000, size(c_train,4));
# random_indices = rand(1:size(c_train,4), num_samples);

# for i in random_indices
#     predicted_c_standardized = generate_closure(velocity_cnn, ps_drift, _st_drift, v_train_standardized[:,:,:,i], num_steps, 1, dev; method=:euler) |> dev;
#     predicted_c_standardized = predicted_c_standardized |> cpu_device();
#     difference = mean(abs2, c_train_standardized[:,:,:,i] .- predicted_c_standardized);
#     global max_difference = max(max_difference, difference);
#     global min_difference = min(min_difference, difference);
# end
# println("The minimum mean squared error in the training set: ", min_difference)
# println("The maximum mean squared error in the training set: ", max_difference)

# ### Time evolution where we start with filtered DNS solution ###
# t = 0.0f0; # Initial time t_0
# dt = 2.0f-4; # Time step Δt
# nt = 250; # Number of time steps (number of training and test samples)

# num_steps = 100;  # Number of steps to evolve the image
# step_size = 1.0 / num_steps;  # Step size (proportional to time step)
# batch_size = 1; 

# # # Start with initial condition
# ubar_test = v_train[:,:,:,1]; 
# ubar_test = reshape(ubar_test, N_les, N_les, 2, batch_size) |> dev; # size: (128,128,2,1) - GPU (if we generate random) or CPU (if we take the initial condition for the training)
# u_les = ubar_test;

# ## TREAT CLOSURE NOT AS A STATE!!
# for i = 1:nt+1
#     global u_les, ubar_test
#     if i > 1
#         global t
#         # At t=2: we know ubar_test for t=1 (Initial Condition)
#         # Compute the RHS closure:
#         stand_ubar_test = inverse_standardize_set_per_channel(ubar_test, state_means, state_std)[:,:,:,1]|> dev
#         stand_closure = generate_closure(velocity_cnn, ps_drift, _st_drift, stand_ubar_test, num_steps, batch_size, dev; method=:euler) |> dev; # size: (128,128,2,1) 
#         closure = inverse_standardize_set_per_channel(stand_closure, closure_means, closure_std) |> dev
#         # Take a step with Forward Euler:
#         next_ubar_test = ubar_test .+ (dt .* f_les(ubar_test, nothing, 0.0)) .+ (dt .* closure)
#         ubar_test = next_ubar_test
#         t += dt
#         println("Finished time step ", i)
#     end

#     if i % 10 == 0 && i > 1
#         # At t=1: u_les = ubar_test = v_train[:,:,:,1]. 
#         t = (i-1) * dt
#         ω_model = Array(INS.vorticity(pad_circular(ubar_test, 1; dims = 1:2), setup_les))[:,:,1]
#         ω_nomodel = Array(INS.vorticity(pad_circular(u_les, 1; dims=1:2), setup_les))[:,:,1]
#         ω_groundtruth =  Array(INS.vorticity(pad_circular(v_train[:,:,:,i] |> dev, 1; dims=1:2), setup_les))[:,:,1]

#         ω_model = ω_model[2:end-1, 2:end-1];
#         ω_nomodel = ω_nomodel[2:end-1, 2:end-1];
#         ω_groundtruth = ω_groundtruth[2:end-1, 2:end-1];

#         ω_closure = abs.(ω_groundtruth - ω_nomodel);
#         ω_pred_closure = abs.(ω_model-ω_nomodel);
#         ω_error_map = abs.(ω_model-ω_groundtruth);

#         title_model = @sprintf("Vorticity model, t = %.3f", t)
#         title_nomodel = @sprintf("Vorticity no model, t=%.3f", t)
#         title_groundtruth = @sprintf("Vorticity ground truth, t=%.3f", t)
#         title_closure = @sprintf("vorticity closure, t=%.3f", t)
#         title_pred_closure = @sprintf("Predicted closure, t=%.3f", t)
#         title_error = @sprintf("Error, t=%.3f", t)

#         p1 = Plots.heatmap(ω_nomodel'; xlabel = "x", ylabel="y", title=title_nomodel, color=:viridis)
#         p2 = Plots.heatmap(ω_model'; xlabel = "x", ylabel = "y", title=title_model, color=:viridis)
#         p3 = Plots.heatmap(ω_groundtruth'; xlabel = "x", ylabel = "y", title=title_groundtruth, color=:viridis)
#         p4 = Plots.heatmap(ω_closure'; xlabel="x", ylabel="y", title=title_closure, color=:viridis)
#         p5 = Plots.heatmap(ω_pred_closure'; xlabel="x", ylabel="y", title=title_pred_closure, color=:viridis)
#         p6 = Plots.heatmap(ω_error_map'; xlabel="x", ylabel="y", title=title_error)
        
#         # Combine both plots into a single figure
#         fig = Plots.plot(p1, p2, p3, p4, p5, p6, layout = (2, 3), size=(2400, 800))
        
#         # Save each plot as an image file
#         savefig(fig, @sprintf("vorticity_timestep_%03d.png", i))

#         # Print the computed error
#         println("Error between model and ground truth at t = ", t, ": ", mean(ω_error_map))
#     end
# end