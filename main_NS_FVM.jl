using .master_project

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
using GLMakie

#### TRAINING ####
dev = cpu_device()
if dev==gpu_device()
    gr()
    ENV[" GKSwstype"] = "100"
    CUDA.allowscalar(false)
end

### Generate data - Incompressible Navier Stokes equations: filtered DNS & LES state & closure terms ### 
Re = 1.0f3; # Set the Reynolds value instead of the viscosity.
num_initial_conditions = 2; # Add the number of initial condition.

# Create two different parameter sets for DNS and LES (K = resolution) K_{LES} < K_{DNS}
N_les = 64; # Grid: 64 x 64
N_dns = 256; # Grid: 256 x 256

t = 0.0f0; # Initial time t_0
dt = 2.0f-4; # Time step Δt
nt = 50; # Number of time steps (number of training and test samples)

create_right_hand_side(setup, psolver) = function right_hand_side(u, p, t)
    u = pad_circular(u, 1; dims = 1:2)
    F = INS.momentum(u, nothing, t, setup)
    F = F[2:end-1, 2:end-1, :]
    F = pad_circular(F, 1; dims = 1:2)

    PF = INS.project(F, setup; psolver)
    stack(PF)[2:end-1, 2:end-1, :]
end
 
# Setup DNS - Grid
x_dns = LinRange(0.0, 1.0, N_dns + 1), LinRange(0.0, 1.0, N_dns + 1);
setup_dns = INS.Setup(; x=x_dns, Re);
# Setup DNS - Initial condition
ustart_dns = INS.random_field(setup_dns, 0.0); # size: (258, 258, 2)
ustart_dns = ustart_dns[2:end-1, 2:end-1, :]; # size: (256, 256, 2)
# Setup DNS - psolver
psolver_dns = INS.psolver_spectral(setup_dns);

# Setup LES - Grid 
x_les = LinRange(0.0, 1.0, N_les+1), LinRange(0.0, 1.0, N_les+1);
setup_les = INS.Setup(; x=x_les, Re);
# Setup LES - psolver
psolver_les = INS.psolver_spectral(setup_les);

# SciML-compatible right hand side function
f_dns = create_right_hand_side(setup_dns, psolver_dns);
f_les = create_right_hand_side(setup_les, psolver_les);

# Initialize empty arrays for concatenating training data
v_train = Array{Float32}[] |> dev;
c_train = Array{Float32}[] |> dev;

# Set paths to save/load data
v_train_path = "v_train_data_FVM.bson";
c_train_path = "c_train_data_FVM.bson";
generate_new_data = true;  # Set this to true if you want to regenerate data

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
        v = zeros(N_les, N_les, 2, nt + 1) |> dev; # Do we need to add an extra dimension
        c = zeros(N_les, N_les, 2, nt + 1) |> dev;

        # Define initial condition for DNS.
        # global u = random_field(params_dns) |> dev;

        global u = INS.random_field(setup_dns, 0.0) |> dev; # size: (258, 258, 2)
        u = u[2:end-1, 2:end-1, :] # size: (256, 256, 2)
        nburn = 500; # number of steps to stabilize the simulation before collecting data.
        # Stabilize initial condition
        for i = 1:nburn
            global u = step_rk4(u, dt, f_dns)
        end

        # Generate time evolution data
        for i = 1:nt+1
            # Update DNS solution at each timestep
            if i > 1
                global t, u
                t += dt
                u = step_rk4(u, dt, f_dns)
            end

            # Compute filtered DNS by using the face-averaging filter 
            ubar = face_averaging_velocity_2D(u, N_dns, N_les); # size: (64, 64, 2)
            v[:, :, :, i] = Array(ubar)

            # Compute the closure term
            filtered_RHS = face_averaging_velocity_2D(f_dns(u, nothing, 0.0), N_dns, N_les)
            RHS_ubar = f_les(ubar, nothing, 0.0)
            c[:, :, :, i] = Array(filtered_RHS - RHS_ubar)
            
            # Generate visualizations every 10 steps
            if i % 10 == 0
                ω_dns = Array(INS.vorticity(pad_circular(u, 1; dims = 1:2), setup_dns))
                ω_les = Array(INS.vorticity(pad_circular(ubar, 1; dims = 1:2), setup_les))

                title_dns = @sprintf("Vorticity (DNS), t = %.3f", t)
                title_les = @sprintf("Vorticity (Filtered DNS), t = %.3f", t)

                p1 = Plots.heatmap(ω_dns'; xlabel = "x", ylabel = "y", title = title_dns, color=:viridis)
                p2 = Plots.heatmap(ω_les'; xlabel = "x", ylabel = "y", title = title_les, color=:viridis)

                fig = Plots.plot(p1, p2, layout = (1, 2), size=(1200, 400))
                frame(anim, fig)
            end
        end

        # Concatenate along the time dimension for all initial conditions
        if cond == 1
            global v_train = Array(v[:,:,:,:]) |> dev
            global c_train = Array(c[:,:,:,:]) |> dev
        else
            global v_train = cat(v_train, Array(v[:,:,:,:]) |> dev; dims=4)
            global c_train = cat(c_train, Array(c[:,:,:,:]) |> dev; dims=4)
        end
    end
    gif(anim, "vorticity_comparison_animation.gif")

    # Save the generated data
    println("Saving generated dataset...")
    serialize(v_train_path, v_train)
    serialize(c_train_path, c_train)
end
println("Dataset ready for use, which is of size: ", size(v_train))

# Compute mean and standard deviation for `v_train` and standardize the set. 
state_means, state_std = compute_mean_std(v_train);
v_train_standardized = standardize_training_set_per_channel(v_train, state_means, state_std);
v_train_standardized = v_train_standardized |> dev;

# Compute mean and standard deviation for `c_train` and standardize the set.
closure_means, closure_std = compute_mean_std(c_train);
c_train_standardized = standardize_training_set_per_channel(c_train, closure_means, closure_std);
c_train_standardized = c_train_standardized |> dev;

# Define the network - u-net with ConvNextBlocks 
velocity_cnn = build_full_unet(16,[32,64,128],8; dev);

# Is the initial data gaussian distributed?
is_gaussian = true; 
# Name of potential new model
model_name = "gaussian_model_FVM";
# Path of potential loaded model. 
load_path = nothing; # "trained_models/gaussian_model_FVM.bson";  # Set to `nothing` to initialize instead

# Load/Initialize parameters of the network - ODE
ps_drift, st_drift, opt_drift = initialize_or_load_model(model_name, velocity_cnn, load_path; dev);

# Define the batch_size, num_batches, num_epochs
batch_size = 32;
num_samples = size(c_train,4);
num_batches = ceil(Int, num_samples / batch_size);
num_epochs = 10;

# Train ODE 
train!(velocity_cnn, ps_drift, st_drift, opt_drift, num_epochs, batch_size, c_train_standardized, v_train, num_batches, dev, model_name, "trained_models");
# # Train SDE
# train!(velocity_cnn, ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser, num_epochs, batch_size, v_train, c_train, v_train, num_batches, dev, is_gaussian, model_name, "trained_models");

#### USE TRAINED MODEL #####;
# # Set to test mode
_st_drift = Lux.testmode(st_drift) |> dev;
# _st_denoiser = Lux.testmode(st_denoiser) |> dev; # (Add when using the SDE)

# Compute the error between the predicted closure and the real closure of a random set of the training data.
num_steps = 20; # For the ODE evaluation. 
max_difference = -Inf;
min_difference = Inf;

num_samples = min(5, size(c_train,4));
random_indices = rand(1:size(c_train,4), num_samples);

for i in random_indices
    predicted_c_standardized = generate_closure(velocity_cnn, ps_drift, _st_drift, v_train_standardized[:,:,:,i], num_steps, 1, dev; method=:euler) |> dev;
    difference = mean(abs2, c_train_standardized[:,:,:,i] .- predicted_c_standardized);
    global max_difference = max(max_difference, difference);
    global min_difference = min(min_difference, difference);
end
println("The minimum mean squared error in the training set: ", min_difference)
println("The maximum mean squared error in the training set: ", max_difference)

### Time evolution where we start with filtered DNS solution ###
t = 0.0f0; # Initial time t_0
dt = 2.0f-4; # Time step Δt
nt = 3; # Number of time steps (number of training and test samples)

num_steps = 100;  # Number of steps to evolve the image
step_size = 1.0 / num_steps;  # Step size (proportional to time step)
batch_size = 1; 

# # Start with initial condition
ubar_test = v_train[:,:,:,1]; 
ubar_test = reshape(ubar_test, N_les, N_les, 2, batch_size) |> dev; # size: (128,128,2,1) - GPU (if we generate random) or CPU (if we take the initial condition for the training)
u_les = ubar_test;

## TREAT CLOSURE NOT AS A STATE!!
for i = 1:nt+1
    global u_les, ubar_test
    if i > 1
        global t
        # At t=2: we know ubar_test for t=1 (Initial Condition)
        # Compute the RHS closure:
        stand_ubar_test = inverse_standardize_set_per_channel(ubar_test, state_means, state_std)[:,:,:,1]|> dev
        stand_closure = generate_closure(velocity_cnn, ps_drift, _st_drift, stand_ubar_test, num_steps, batch_size, dev; method=:euler) |> dev; # size: (128,128,2,1) 
        closure = inverse_standardize_set_per_channel(stand_closure, closure_means, closure_std) |> dev
        # Take a step with Forward Euler:
        next_ubar_test = ubar_test .+ (dt .* f_les(ubar_test, nothing, 0.0)) .+ (dt .* closure)
        ubar_test = next_ubar_test
        t += dt
        println("Finished time step ", i)
    end

    if i % 1 == 0 && i > 1
        # At t=1: u_les = ubar_test = v_train[:,:,:,1]. 
        t = (i-1) * dt
        ω_model = Array(INS.vorticity(pad_circular(ubar_test, 1; dims = 1:2), setup_les))[:,:,1]
        ω_nomodel = Array(INS.vorticity(pad_circular(u_les, 1; dims=1:2), setup_les))[:,:,1]
        ω_groundtruth =  Array(INS.vorticity(pad_circular(v_train[:,:,:,i], 1; dims=1:2), setup_les))[:,:,1]

        error_map = abs.(ω_model-ω_groundtruth)

        title_model = @sprintf("Vorticity model, t = %.3f", t)
        title_nomodel = @sprintf("Vorticity no model, t=%.3f", t)
        title_groundtruth = @sprintf("Vorticity ground truth, t=%.3f", t)

        p1 = Plots.heatmap(ω_nomodel'; xlabel = "x", ylabel="y", title=title_nomodel, color=:viridis)
        p2 = Plots.heatmap(ω_model'; xlabel = "x", ylabel = "y", title=title_model, color=:viridis)
        p3 = Plots.heatmap(ω_groundtruth'; xlabel = "x", ylabel = "y", title=title_groundtruth, color=:viridis)
        
        # Combine both plots into a single figure
        fig = Plots.plot(p1, p2, p3, layout = (1, 3), size=(1200, 400))
        
        # Save each plot as an image file
        savefig(fig, @sprintf("vorticity_timestep_%03d.png", i))

        # Print the computed error
        println("Error between model and ground truth at t = ", t, ": ", sum(error_map))
    end
end