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
using IncompressibleNavierStokes 
const INS = IncompressibleNavierStokes
using OrdinaryDiffEq
using GLMakie

#### TRAINING ####
# Choose device: CPU or GPU
dev = cpu_device()
CUDA.allowscalar(false)



### Generate data - Incompressible Navier Stokes equations: filtered DNS & LES state & closure terms ### 
Re = 1.0f3; # Set the parameter value ν.
num_initial_conditions = 5; # Add the number of initial condition

# Create two different parameter sets for DNS and LES (K = resolution) K_{LES} < K_{DNS}
N_les = 32; # Grid: 32 x 32
N_dns = 128; # Grid: 128 x 128

t = 0.0f0; # Initial time t_0
dt = 2.0f-4; # Time step Δt
nt = 5000; # Number of time steps (number of training and test samples)
create_right_hand_side(setup, psolver) = function right_hand_side(u, p, t)
    # u = eachslice(u; dims = ndims(u))
    # u = (u...,)
    # u = stack(u)
    # u = INS.apply_bc_u(u, t, setup)
    u = pad_circular(u, 1; dims = 1:2)
    F = INS.momentum(u, nothing, t, setup)
    F = F[2:end-1, 2:end-1, :]
    # F = INS.apply_bc_u(F, t, setup; dudt = true)
    F = pad_circular(F, 1; dims = 1:2)

    PF = INS.project(F, setup; psolver)
    stack(PF)[2:end-1, 2:end-1, :]
end


# Setup
x = LinRange(0.0, 1.0, N_dns + 1), LinRange(0.0, 1.0, N_dns + 1);
setup = INS.Setup(; x, Re);
ustart = INS.random_field(setup, 0.0);
ustart = ustart[2:end-1, 2:end-1, :]
psolver = INS.psolver_spectral(setup);

x_les = LinRange(0.0, 1.0, N_les+1), LinRange(0.0, 1.0, N_les+1);
setup_les = INS.Setup(; x=x_les, Re);
psolver_les = INS.psolver_spectral(setup_les);

# SciML-compatible right hand side function
# Note: Requires `stack(u)` to create one array
f = create_right_hand_side(setup, psolver);
lol = f(ustart, nothing, 0.0)

# Solve the ODE using SciML
prob = ODEProblem(f, stack(ustart), (0.0, 1.0))
sol = solve(
    prob,
    Tsit5();
    # adaptive = false,
    dt = 1e-4,
)
sol.t

# # Animate solution
# let
#     (; Iu) = setup.grid
#     i = 1
#     obs = sqrt.(sol.u[1][:, :, 1].^2 .+ sol.u[1][:, :, 2].^2)

#     # obs[] = s

#     fig = INS.heatmap(obs)
#     fig |> display
#     for u in sol.u
#         obs = sqrt.(u[:, :, 1].^2 .+ u[:, :, 2].^2)
#         # obs = 
#         # fig |> display
#         sleep(0.05)
#     end
# end


# Initialize empty arrays for concatenating training data
v_train = Array{Float32}[] |> dev
c_train = Array{Float32}[] |> dev

# Set paths to save/load data
v_train_path = "v_train_data.bson"
c_train_path = "c_train_data.bson"
generate_new_data = true  # Set this to true if you want to regenerate data

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
        v = zeros(N_les, N_les, 2, nt + 1) |> dev;
        c = zeros(N_les, N_les, 2, nt + 1) |> dev;

        # Define initial condition for DNS.
        # global u = random_field(params_dns) |> dev;

        global u = INS.random_field(setup, 0.0) |> dev;
        u = u[2:end-1, 2:end-1, :]
        nburn = 500; # number of steps to stabilize the simulation before collecting data.


        # Stabilize initial condition
        for i = 1:nburn
            global u
            u = step_rk4(u, dt, f)
        end

        # Generate time evolution data
        for i = 1:nt+1
            # Update DNS solution at each timestep
            if i > 1
                global t, u
                t += dt
                u = step_rk4(u, dt, f)
            end

            # Compute filtered DNS and closure term
            # ubar = spectral_cutoff(u, params_les.K)
            ubar = u[2:4:end-1, 2:4:end-1, :]
            v[:, :, :, i] = Array(ubar)
            c[:, :, :, i] = Array(ubar)# Array(spectral_cutoff(F(u, params_dns), params_les.K) - F(ubar, params_les))

            
            # Generate visualizations every 10 steps
            if i % 100 == 0
                
                ω_dns = Array(INS.vorticity(pad_circular(u, 1; dims = 1:2), setup))
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
is_gaussian = true; 
# Name of potential new model
model_name = "gaussian_model";
# Path of potential loaded model. 
load_path = "trained_models/gaussian_model.bson";  # Set to `nothing` to initialize instead

# Call the function
ps_drift, st_drift, opt_drift, model_name = initialize_or_load_model(model_name, is_gaussian, velocity_cnn, dev, load_path);

# Define the batch_size, num_batches, num_epochs
batch_size = 32;
num_samples = size(c_train,4);
num_batches = ceil(Int, num_samples / batch_size);
num_epochs = 10;

# Train ODE 
train!(velocity_cnn, ps_drift, st_drift, opt_drift, num_epochs, batch_size, c_train, v_train, num_batches, dev, model_name, "trained_models");
# # Train SDE
# train!(velocity_cnn, ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser, num_epochs, batch_size, v_train, c_train, v_train, num_batches, dev, is_gaussian, model_name, "trained_models");

#### USE TRAINED MODEL #####;
# # Set to test mode
_st_drift = Lux.testmode(st_drift) |> dev;
# _st_denoiser = Lux.testmode(st_denoiser) |> dev;

# closure error 
mean_error = compute_mean_error(velocity_cnn, ps_drift, _st_drift, v_train, c_train, dev)

# mean_error = compute_mean_error(velocity_cnn, ps_drift, st_drift, ps_denoiser, st_denoiser, v_train, c_train, v_train, dev, is_gaussian)

# ### Time evolution where we start with filtered DNS solution ###
# t = 0.0f0; # Initial time t_0
# dt = 2.0f-4; # Time step Δt
# nt = 3; # Number of time steps (number of training and test samples)

# num_steps = 100;  # Number of steps to evolve the image
# step_size = 1.0 / num_steps;  # Step size (proportional to time step)
# batch_size = 1; 

# # # Start with initial condition
# ubar_test = v_train[:,:,:,1]; 
# ubar_test = reshape(ubar_test, 32, 32, 2, batch_size) |> dev; # size: (128,128,2,1) - GPU (if we generate random) or CPU (if we take the initial condition for the training)
# u_les = ubar_test;

# ## TREAT CLOSURE NOT AS A STATE!!
# for i = 1:nt+1
#     global u_les, ubar_test
#     if i > 1
#         global t
#         u_les = step_rk4(ubar_test, params_les, dt) |> dev
#         closure = generate_closure(ubar_test, ubar_test, batch_size, step_size, ps_drift, _st_drift, ps_denoiser, _st_denoiser, velocity_cnn, dev, is_gaussian) |> dev # size: (128,128,2,1) 
#         # params_with_closure = merge(params_les, Dict(:m => closure))

#         # Perform element-wise addition for the next ubar_test
#         ubar_test = u_les .+ (dt .* closure) |> dev # size: (128,128,2,1) - CPU
#         t += dt
#         println("Finished time step ", i)
#     end

#     # Generate plots for each time step and save as an image
#     if i % 1 == 0
#         t = (i - 1) * dt
#         ω_model = Array(vorticity(ubar_test, params_les))[:,:,1] # size: (128,128)
#         ω_nomodel = Array(vorticity(CuArray(u_les), params_les))[:,:,1] # size: (128,128)
#         ω_groundtruth = Array(vorticity(CuArray(v_train[:,:,:,i]), params_les))[:,:,1] 
        
#         error_map = abs.(ω_model-ω_groundtruth)
        
#         title_model = @sprintf("Vorticity model, t = %.3f", t)
#         title_nomodel = @sprintf("Vorticity no model, t=%.3f", t)
#         title_groundtruth = @sprintf("Vorticity ground truth, t=%.3f", t)
#         p1 = heatmap(ω_nomodel'; xlabel = "x", ylabel="y", title=title_nomodel)
#         p2 = heatmap(ω_model'; xlabel = "x", ylabel = "y", title=title_model)
#         p3 = heatmap(ω_groundtruth'; xlabel = "x", ylabel = "y", title=title_groundtruth)
        
#         # Combine both plots into a single figure
#         fig = plot(p1, p2, p3, layout = (1, 3), size=(1200, 400))
        
#         # Save each plot as an image file
#         savefig(fig, @sprintf("vorticity_timestep_%03d.png", i))

#         # Print the computed error
#         println("Error between model and ground truth at t = ", t, ": ", sum(error_map))
#     end
# end

# # anim = Animation()
# # for i = 1:nt+1
# #     global u_les, ubar_test
# #     if i > 1
# #         global t
# #         u_les = step_rk4(ubar_test, params_les, dt) |> dev # size: (128,128,2,1) - GPU
# #         closure = generate_closure(ubar_test, ubar_test, batch_size, step_size, ps_drift, _st_drift, ps_denoiser, _st_denoiser, velocity_cnn, dev, is_gaussian) |> dev # size: (128,128,2,1) - CPU
# #         # Perform element-wise addition for the next ubar_test
# #         ubar_test = u_les .+ closure |> dev # size: (128,128,2,1) - CPU
# #         t += dt
# #         println("Finished time step ", i)
# #     end

# #     if i % 1 == 0
# #         t = (i - 1) * dt
# #         ω_model = Array(vorticity(ubar_test, params_les))[:,:,1] # size: (128,128)
# #         ω_nomodel = Array(vorticity(CuArray(u_les), params_les))[:,:,1] # size: (128,128)
# #         title_model = @sprintf("Vorticity model, t = %.3f", t)
# #         title_nomodel = @sprintf("Vorticity no model, t=%.3f", t)
# #         p1 = heatmap(ω_nomodel'; xlabel = "x", ylabel="y", title=title_nomodel)
# #         p2 = heatmap(ω_model'; xlabel = "x", ylabel = "y", title = title_model)
# #         fig = plot(p1, p2, layout = (1, 2), size=(800, 400))  # Combine both plots
# #         frame(anim, fig)  # Add frame to animation
# #     end
# # end
# # gif(anim, "voritcity_closuremodel.gif")  