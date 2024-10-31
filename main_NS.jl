using master_project

using Lux
using Random
using ComponentArrays
using Optimisers
using Statistics
using Plots
using CUDA
using Printf

# Choose CPU or GPU
dev = cpu_device() # gpu_device()

### Generate data - Incompressible Navier Stokes equations: filtered DNS & LES state & closure terms ### 
nu = 5.0f-4; # Set the parameter value ν.

# Create two different parameter sets for DNS and LES (K = resolution) K_{LES} < K_{DNS}
params_les = create_params(32; nu); # Grid: 64 x 64
params_dns = create_params(64; nu); # Grid: 128 x 128

t = 0.0f0; # Initial time t_0
dt = 2.0f-4; # Time step Δt
nt = 1000; # Number of time steps (number of training and test samples)

v = zeros(Complex{Float32}, params_les.N, params_les.N, 2, nt + 1); # Array for Filtered DNS solution. 
c = zeros(Complex{Float32}, params_les.N, params_les.N, 2, nt + 1); # Array for closure term. 

# Define initial condition for DNS. 
u = random_field(params_dns);
nburn = 500; # number of steps to stabilize the simulation before collecting data, creating a stable/realistic initial condition.
# create the initial condition.
for i = 1:nburn
    u = step_rk4(u, params_dns, dt)
end 

# Filter: Chop off frequencies, retaining frequencies up to K, and multiply with scaling factor related to the size of the grid. 
spectral_cutoff(u, K) = (2K)^2 / (size(u, 1) * size(u, 2)) * [
    u[1:K, 1:K, :] u[1:K, end-K+1:end, :]
    u[end-K+1:end, 1:K, :] u[end-K+1:end, end-K+1:end, :]
] 


# Time stepping - Generating training&test data set: Filtered DNS (Initial distribution) - Closure term (Target distribution)
anim = Animation()
for i = 1:nt+1
    # First compute the filtered DNS & closure term for the initial condition.
    if i > 1  
        t += dt
        u = step_rk4(u, params_dns, dt) # DNS solution
    end 
    ubar = spectral_cutoff(u, params_les.K) # filtered DNS solution.
    v[:, :, :, i] = Array(ubar) # Add u_bar to v. 
    c[:, :, :, i] = Array(spectral_cutoff(F(u, params_dns), params_les.K) - F(ubar, params_les)) # closure term. 
    if i % 10 == 0
        ω_dns = Array(vorticity(u, params_dns))
        ω_les = Array(vorticity(ubar, params_les))

        title_dns = @sprintf("Vorticity (DNS), t = %.3f", t)
        title_les = @sprintf("Vorticity (Filtered DNS), t = %.3f", t)

        # Create side-by-side plots of DNS and filtered DNS vorticity
        p1 = heatmap(ω_dns'; xlabel = "x", ylabel = "y", title = title_dns, color=:viridis)
        p2 = heatmap(ω_les'; xlabel = "x", ylabel = "y", title = title_les, color=:viridis)

        fig = plot(p1, p2, layout = (1, 2), size=(1200, 400))  # Combine both plots
        frame(anim, fig)  # Add frame to animation
    end
end
gif(anim, "voritcity_comparison_animation.gif")

# Transform the v and c to set as correct pairs for the training process.
v_train = v[:,:,:,1:nt];
c_train = c[:,:,:,2:nt+1];

# Define the network - u-net with ConvNextBlocks 
velocity_cnn = build_full_unet(16,[32,64,128],8);

# Initialize the network parameters and the state - ODE
ps_drift, st_drift = Lux.setup(Random.default_rng(), velocity_cnn); # .|> dev;
# Initialize the network parameters and the states (drift and score) - SDE
ps_denoiser, st_denoiser = Lux.setup(Random.default_rng(), velocity_cnn); # |> dev;

# Define the batch_size, num_batches, num_epochs
batch_size = 32;
num_samples = size(c_train,4);
num_batches = ceil(Int, num_samples / batch_size);
num_epochs = 500;

# Define the Adam optimizer with a learning rate 
opt_drift = Optimisers.setup(Adam(1.0e-3, (0.9f0, 0.99f0), 1e-10), ps_drift);
opt_denoiser = Optimisers.setup(Adam(1.0e-3, (0.9f0, 0.99f0), 1e-10), ps_denoiser);

# Is the initial data gaussian distributed?
is_gaussian = false;

# Start training.
train!(velocity_cnn, ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser, num_epochs, batch_size, v_train, c_train, v_train, num_batches, dev, is_gaussian, "trained_models");

# # Load the model for generating the closure term after closing the program. 
# ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser = load_model("trained_models/final_model.bson");

# Set to test mode
_st_drift = Lux.testmode(st_drift);
_st_denoiser = Lux.testmode(st_denoiser);

### Time evolution where we start with filtered DNS solution ###
t = 0.0f0; # Initial time t_0
dt = 2.0f-4; # Time step Δt
nt = 20; # Number of time steps (number of training and test samples)

num_steps = 100;  # Number of steps to evolve the image
step_size = 1.0 / num_steps;  # Step size (proportional to time step)
batch_size = 1; 

# Start with initial conditio
# # Define initial condition for DNS. 
# u_test = random_field(params_dns);
# nburn = 500; 
# for i = 1:nburn
#     u_test = step_rk4(u_test, params_dns, dt); # size: (128,128,2,1) - GPU
# end 

# # Define the filtered DNS initial condition. 
ubar_test = v[:,:,:,1]; # To check if the network works use the same initial condition as the training set. 
# ubar_test = spectral_cutoff(u_test, params_les.K);
ubar_test = reshape(ubar_test, 64, 64, 2, batch_size); # size: (128,128,2,1) - GPU (if we generate random) or CPU (if we take the initial condition for the training)
ubar_test = CuArray(ubar_test);
u_les = ubar_test;

anim = Animation()
for i = 1:nt+1
    global u_les, ubar_test
    if i > 1
        u_les = step_rk4(ubar_test, params_les, dt) # size: (128,128,2,1) - GPU
        u_les = Array(u_les) # size: (128,128,2,1) - CPU
        closure = generate_closure(ubar_test, ubar_test, batch_size, step_size, ps_drift, _st_drift, ps_denoiser, _st_denoiser, velocity_cnn, dev, is_gaussian) # size: (128,128,2,1) - CPU
        # Perform element-wise addition for the next ubar_test
        ubar_test = u_les .+ closure # size: (128,128,2,1) - CPU
        ubar_test = CuArray(ubar_test) # size: (128,128,2,1) - GPU
        t += dt
        println("Finished time step ", i)
    end

    if i % 2 == 0
        t = (i - 1) * dt
        ω_model = Array(vorticity(ubar_test, params_les))[:,:,1] # size: (128,128)
        ω_nomodel = Array(vorticity(CuArray(u_les), params_les))[:,:,1] # size: (128,128)
        title_model = @sprintf("Vorticity model, t = %.3f", t)
        title_nomodel = @sprintf("Vorticity no model, t=%.3f", t)
        p1 = heatmap(ω_nomodel'; xlabel = "x", ylabel="y", title=title_nomodel)
        p2 = heatmap(ω_model'; xlabel = "x", ylabel = "y", title = title_model)
        fig = plot(p1, p2, layout = (1, 2), size=(800, 400))  # Combine both plots
        frame(anim, fig)  # Add frame to animation
    end
end
gif(anim, "voritcity_closuremodel.gif")  