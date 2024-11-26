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

# Choose CPU or GPU
dev = gpu_device() # gpu_device()
if dev==gpu_device()
    gr()
    ENV["GKSwstype"] = "100"
    CUDA.allowscalar(false)
end

### Generate data - Incompressible Navier Stokes equations: filtered DNS & LES state & closure terms ### 
nu = 5.0f-4; # Set the parameter value ν.
num_initial_conditions = 5;

# Create two different parameter sets for DNS and LES (K = resolution) K_{LES} < K_{DNS}
params_les = create_params(32; nu); # Grid: 64 x 64
params_dns = create_params(128; nu); # Grid: 256 x 256

t = 0.0f0; # Initial time t_0
dt = 2.0f-4; # Time step Δt
nt = 5000; # Number of time steps (number of training and test samples)

spectral_cutoff(u,K) = (2K)^2/(size(u,1) * size(u,2)) * [
    u[1:K, 1:K, :] u[1:K, end-K+1:end, :]
    u[end-K+1:end, 1:K, :] u[end-K+1:end, end-K+1:end, :]
]

# Initialize empty arrays for concatenating training data
v_train = Complex{Float32}[] |> dev;
c_train = Complex{Float32}[] |> dev;

# Set paths to save/load data
v_train_path = "v_train_data.bson";
c_train_path = "c_train_data.bson";
generate_new_data = false;  # Set this to true if you want to regenerate data

if !generate_new_data && isfile(v_train_path) && isfile(c_train_path)
    # Load existing data if available
    println("Loading existing dataset...")
    v_train = deserialize(v_train_path) |> dev
    c_train = deserialize(c_train_path) |> dev
else
    println("Generating new dataset...")

    anim = Animation()
    for cond = 1:num_initial_conditions
        println("Generating data for initial condition $cond")

        # GPU version
        v = zeros(Complex{Float32}, params_les.N, params_les.N, 2, nt + 1) |> dev;
        c = zeros(Complex{Float32}, params_les.N, params_les.N, 2, nt + 1) |> dev;

        # Define initial condition for DNS.
        global u = random_field(params_dns);
        nburn = 500; # number of steps to stabilize the simulation before collecting data.

        # Stabilize initial condition
        for i = 1:nburn
            global u
            u = step_rk4(u, params_dns, dt)
        end

        # Generate time evolution data
        for i = 1:nt+1
            # Update DNS solution at each timestep
            if i > 1
                global t, u
                t += dt
                u = step_rk4(u, params_dns, dt)
            end
        
            # Compute filtered DNS and closure term
            ubar = spectral_cutoff(u, params_les.K)
            v[:, :, :, i] = Array(ubar)
            c[:, :, :, i] = Array(spectral_cutoff(F(u, params_dns), params_les.K) - F(ubar, params_les))
        
            # Generate visualizations every 10 steps
            if i % 100 == 0
                ω_dns = Array(vorticity(u, params_dns))
                ω_les = Array(vorticity(ubar, params_les))

                title_dns = @sprintf("Vorticity (DNS), t = %.3f", t)
                title_les = @sprintf("Vorticity (Filtered DNS), t = %.3f", t)

                p1 = heatmap(ω_dns'; xlabel = "x", ylabel = "y", title = title_dns, color=:viridis)
                p2 = heatmap(ω_les'; xlabel = "x", ylabel = "y", title = title_les, color=:viridis)

                fig = plot(p1, p2, layout = (1, 2), size=(1200, 400))
                frame(anim, fig)
            end
        end

        # Concatenate along the time dimension for all initial conditions
        if cond == 1
            global v_train = Array(v[:,:,:,1:nt]) |> dev
            global c_train = Array(c[:,:,:,2:nt+1]) |> dev
        else
            global v_train = cat(v_train, Array(v[:,:,:,1:nt]) |> dev; dims=4)
            global c_train = cat(c_train, Array(c[:,:,:,2:nt+1]) |> dev; dims=4)
        end
    end
    gif(anim, "vorticity_comparison_animation_spectral.gif")

    # Save the generated data
    println("Saving generated dataset...")
    serialize(v_train_path, v_train)
    serialize(c_train_path, c_train)
end
println("Dataset ready for use, which is of size: ", size(v_train))

# Standardize your data: state and closure per channel
v_train_standardized, state_means, state_std = standardize_training_set_per_channel(v_train);
v_train_standardized = v_train_standardized |> dev;
c_train_standardized, closure_means, closure_std = standardize_training_set_per_channel(c_train);
c_train_standardized = c_train_standardized |> dev;

# Define the network - u-net with ConvNextBlocks 
velocity_cnn = build_full_unet(16,[32,64,128],8; dev);

# Is the initial data gaussian distributed?
is_gaussian = true;
# Name of the (new/loaded) model
model_name = "gaussian_model";
# Path of loaded model
load_path = "trained_models/gaussian_model.bson"; # Set to nothing if you want to initialize new parameters.

# Load/initialize the network parameters.
ps_drift, st_drift, opt_drift = initialize_or_load_model(model_name, is_gaussian, velocity_cnn, load_path; dev);

# Define the batch_size, num_batches, num_epochs
batch_size = 32;
num_samples = size(c_train,4);
num_batches = ceil(Int, num_samples / batch_size);
num_epochs = 50;

# Start training - ODE 
train!(velocity_cnn, ps_drift, st_drift, opt_drift, num_epochs, batch_size, c_train_standardized, v_train_standardized, num_batches, dev, model_name, "trained_models");
# # Start training - SDE
# train!(velocity_cnn, ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser, num_epochs, batch_size, v_train, c_train, v_train, num_batches, dev, is_gaussian, "trained_models");

#### INFERENCE ####
# Set to test mode
_st_drift = Lux.testmode(st_drift) |> dev;
# _st_denoiser = Lux.testmode(st_denoiser) |> dev; # Add if you use the SDE

# Compute the mean squared error between the real closure and the predicted closure. 
num_steps = 20;
max_difference = -Inf;
min_difference = Inf;

# Get 1000 random indices from the dataset
num_samples = min(1000, size(c_train,4));
random_indices = rand(1:size(c_train, 4), num_samples);

for i in random_indices
    predicted_c_standardized = generate_closure(velocity_cnn, ps_drift, _st_drift, v_train_standardized[:,:,:,i], num_steps, 1, dev; method=:euler)|> dev;
    difference = mean(abs2, c_train_standardized[:, :, :, i] .- predicted_c_standardized);
    global max_difference = max(max_difference, difference);
    global min_difference = min(min_difference, difference);
end
println("The maximum mean squared error in the training set: ", max_difference)
println("The minimum mean squared error in the training set: ", min_difference)

### Time evolution where we start with filtered DNS solution ###
# Physical time 
t = 0.0f0; # Initial time t_0
dt = 2.0f-4; # Time step Δt
nt = 6; # Number of time steps (number of training and test samples)

ubar_test = v_train[:,:,:,1]; 
ubar_test = reshape(ubar_test, 2*params_les.K, 2*params_les.K, 2, 1) |> dev;
u_les = ubar_test;

# ODE-pseudo time 
num_steps = 100;  # Number of steps to evolve the image
step_size = 1.0 / num_steps;  # Step size (proportional to time step)
batch_size = 1; 

for i = 1:nt+1
    global u_les, ubar_test
    if i > 1
        global t
        u_les = step_rk4(CuArray(ubar_test), params_les, dt) |> dev # RK4 method to get u_{LES}
        closure = generate_closure(velocity_cnn, ps_drift, _st_drift, v_train_standardized[:,:,:,i], num_steps, batch_size, dev; method=:euler) |> dev; 
        closure = inverse_standardize_per_channel(closure, closure_means, closure_std; dev)
        # Add the closure part using the Forward Euler method to get the approximated ubar. 
        ubar_test = u_les .+ (dt .* closure) |> dev # size: (128,128,2,1) - CPU
        t += dt
        println("Finished time step ", i)
    end

    # Generate plots for each time step and save as an image
    if i % 2 == 0
        t = (i - 1) * dt
        ω_model = Array(vorticity(CuArray(ubar_test), params_les))[:,:,1] 
        ω_nomodel = Array(vorticity(CuArray(u_les), params_les))[:,:,1] 
        ω_groundtruth = Array(vorticity(CuArray(v_train[:,:,:,i]), params_les))[:,:,1] 
        
        error_map = abs.(ω_model-ω_groundtruth)
        
        title_model = @sprintf("Vorticity model, t = %.3f", t)
        title_nomodel = @sprintf("Vorticity no model, t=%.3f", t)
        title_groundtruth = @sprintf("Vorticity ground truth, t=%.3f", t)
        p1 = heatmap(ω_nomodel'; xlabel = "x", ylabel="y", title=title_nomodel)
        p2 = heatmap(ω_model'; xlabel = "x", ylabel = "y", title=title_model)
        p3 = heatmap(ω_groundtruth'; xlabel = "x", ylabel = "y", title=title_groundtruth)
        
        # Combine both plots into a single figure
        fig = plot(p1, p2, p3, layout = (1, 3), size=(1200, 400))
        
        # Save each plot as an image file
        savefig(fig, @sprintf("vorticity_timestep_%03d.png", i))

        # Print the computed error
        println("Error between model and ground truth at t = ", t, ": ", sum(error_map))
    end
end
