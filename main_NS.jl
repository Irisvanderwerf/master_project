using master_project

using Lux
using Random
using ComponentArrays
using Optimisers
using Statistics
using Plots
gr()
ENV["GKSwstype"] = "100"  # Avoids display issues in headless mode
using LuxCUDA
using Printf
using CUDA
using Serialization  # For binary serialization of arrays
using FileIO  # Optional: For using .jld2 file format if you prefer HDF5-style

#### TRAINING ####
# Choose device: CPU or GPU
dev = gpu_device()
CUDA.allowscalar(false) 

### Generate data - Incompressible Navier Stokes equations: filtered DNS & LES state & closure terms ### 
nu = 5.0f-4; # Set the parameter value ν.
num_initial_conditions = 5; # Add the number of initial condition

# Create two different parameter sets for DNS and LES (K = resolution) K_{LES} < K_{DNS}
params_les = create_params(32; nu); # Grid: 64 x 64
params_dns = create_params(128; nu); # Grid: 256 x 256

t = 0.0f0; # Initial time t_0
dt = 2.0f-4; # Time step Δt
nt = 5000; # Number of time steps (number of training and test samples)

# Filter: Chop off frequencies, retaining frequencies up to K, and multiply with scaling factor related to the size of the grid. 
spectral_cutoff(u, K) = (2K)^2 / (size(u, 1) * size(u, 2)) * [
    u[1:K, 1:K, :] u[1:K, end-K+1:end, :]
    u[end-K+1:end, 1:K, :] u[end-K+1:end, end-K+1:end, :]
] 

# Initialize empty arrays for concatenating training data
v_train = Complex{Float32}[] |> dev
c_train = Complex{Float32}[] |> dev

# Set paths to save/load data
v_train_path = "v_train_data.bson"
c_train_path = "c_train_data.bson"
generate_new_data = false  # Set this to true if you want to regenerate data

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
        global u = random_field(params_dns) |> dev;
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
    gif(anim, "vorticity_comparison_animation.gif")

    # Save the generated data
    println("Saving generated dataset...")
    serialize(v_train_path, v_train)
    serialize(c_train_path, c_train)
end
println("Dataset ready for use.")

# Define the network - u-net with ConvNextBlocks 
velocity_cnn = build_full_unet(16,[32,64,128],8);

# Is the initial data gaussian distributed?
is_gaussian = false; 
# Name of potential new model
model_name = "complex_final_model";
# Path of potential loaded model. 
load_path = "trained_models/complex_final_model.bson";  # Set to `nothing` to initialize instead

# Call the function
ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser, model_name = initialize_or_load_model(model_name, is_gaussian, velocity_cnn, dev, load_path);

# # Define the batch_size, num_batches, num_epochs
batch_size = 32;
num_samples = size(c_train,4);
num_batches = ceil(Int, num_samples / batch_size);
num_epochs = 20;

# Start training.
train!(velocity_cnn, ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser, num_epochs, batch_size, v_train, c_train, v_train, num_batches, dev, is_gaussian, model_name, "trained_models");

#### USE TRAINED MODEL #####;

# Set to test mode
_st_drift = Lux.testmode(st_drift) |> dev;
_st_denoiser = Lux.testmode(st_denoiser) |> dev;

# closure error 
mean_error = compute_mean_error(velocity_cnn, ps_drift, st_drift, ps_denoiser, st_denoiser, v_train, c_train, v_train, dev, is_gaussian)

# ### Time evolution where we start with filtered DNS solution ###
# t = 0.0f0; # Initial time t_0
# dt = 2.0f-4; # Time step Δt
# nt = 10; # Number of time steps (number of training and test samples)

# num_steps = 100;  # Number of steps to evolve the image
# step_size = 1.0 / num_steps;  # Step size (proportional to time step)
# batch_size = 1; 

# # Start with initial condition
# ubar_test = v_train[:,:,:,1]; 
# ubar_test = reshape(ubar_test, 64, 64, 2, batch_size) |> dev; # size: (128,128,2,1) - GPU (if we generate random) or CPU (if we take the initial condition for the training)
# u_les = ubar_test;

# anim = Animation()
# for i = 1:nt+1
#     global u_les, ubar_test
#     if i > 1
#         global t
#         u_les = step_rk4(ubar_test, params_les, dt) |> dev # size: (128,128,2,1) - GPU
#         closure = generate_closure(ubar_test, ubar_test, batch_size, step_size, ps_drift, _st_drift, ps_denoiser, _st_denoiser, velocity_cnn, dev, is_gaussian) |> dev # size: (128,128,2,1) - CPU
#         # Perform element-wise addition for the next ubar_test
#         ubar_test = u_les .+ closure |> dev # size: (128,128,2,1) - CPU
#         t += dt
#         println("Finished time step ", i)
#     end

#     if i % 1 == 0
#         t = (i - 1) * dt
#         ω_model = Array(vorticity(ubar_test, params_les))[:,:,1] # size: (128,128)
#         ω_nomodel = Array(vorticity(CuArray(u_les), params_les))[:,:,1] # size: (128,128)
#         title_model = @sprintf("Vorticity model, t = %.3f", t)
#         title_nomodel = @sprintf("Vorticity no model, t=%.3f", t)
#         p1 = heatmap(ω_nomodel'; xlabel = "x", ylabel="y", title=title_nomodel)
#         p2 = heatmap(ω_model'; xlabel = "x", ylabel = "y", title = title_model)
#         fig = plot(p1, p2, layout = (1, 2), size=(800, 400))  # Combine both plots
#         frame(anim, fig)  # Add frame to animation
#     end
# end
# gif(anim, "voritcity_closuremodel.gif")  