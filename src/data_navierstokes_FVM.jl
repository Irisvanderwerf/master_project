using FFTW
using LinearAlgebra
using Random
using CUDA
using ComponentArrays
using IJulia
using Lux
using LuxCUDA
using NNlib
using Optimisers
using Plots
using Random
using Zygote
using KernelAbstractions 
using BSON
using Printf
using Serialization

using IncompressibleNavierStokes
const INS = IncompressibleNavierStokes
using OrdinaryDiffEq

z = CUDA.functional() ? CUDA.zeros : (s...) -> zeros(Float32, s...)
ArrayType = CUDA.functional() ? CuArray : Array
CUDA.allowscalar(false)

# The following function performsone RK4 time step. 
function step_rk4(u0, dt, F)
    a = (
        (0.5f0,),
        (0.0f0, 0.5f0),
        (0.0f0, 0.0f0, 1.0f0),
        (1.0f0 / 6.0f0, 2.0f0 / 6.0f0, 2.0f0 / 6.0f0, 1.0f0 / 6.0f0),
    )
    u = u0
    k = ()
    for i = 1:length(a)
        ki = F(u, nothing, 0.0)
        k = (k..., ki)
        u = u0
        for j = 1:i
            u = u .+ dt .* a[i][j] .* k[j]
        end
    end
    u
end

face_average_syver(u, setup_les, comp) = face_average_syver!(INS.vectorfield(setup_les), u, setup_les, comp)

function face_average_syver!(v, u, setup_les, comp)
    (; grid, backend, workgroupsize) = setup_les
    (; dimension, Nu, Iu) = grid
    D = dimension()
    @kernel function Φ!(v, u, ::Val{α}, face, I0) where {α}
        I = @index(Global, Cartesian)
        J = I0 + comp * (I - oneunit(I))
        s = zero(eltype(v))
        for i in face
            s += u[J+i, α]
        end
        v[I0+I, α] = s / comp^(D - 1)
    end
    for α = 1:D
        ndrange = Nu[α]
        I0 = getoffset(Iu[α])
        face = CartesianIndices(ntuple(β -> β == α ? (comp:comp) : (1:comp), D))
        Φ!(backend, workgroupsize)(v, u, Val(α), face, I0; ndrange)
    end
    v
end

# Function to compute mean and standard deviation per channel
function compute_mean_std(training_set)
    means = [mean(training_set[:,:,c,:,:]) for c in 1:2]
    stds = [std(training_set[:,:,c,:,:]) for c in 1:2]
    return means, stds
end

# Function to standardize the training set per channel
function standardize_training_set_per_channel(training_set, means, stds; one_trajectory=false)
    standardized_set = similar(training_set)
    if !one_trajectory 
        for c in 1:2
            standardized_set[:,:,c,:,:] .= (training_set[:,:,c,:,:] .- means[c]) ./ stds[c]
        end
        return standardized_set
    else
        for c in 1:2
            standardized_set[:,:,c,:] .= (training_set[:,:,c,:] .- means[c]) ./ stds[c]
        end
        return standardized_set
    end
end

# Function of the inverse standardization per channel
function inverse_standardize_set_per_channel(training_set, means, stds; one_trajectory=false)
    inverse_standardized_set = similar(training_set)
    if !one_trajectory
        for c in 1:2
            inverse_standardized_set[:,:,c,:,:] .= (training_set[:,:,c,:,:] .* stds[c]) .+ means[c]
        end
        return inverse_standardized_set
    else
        for c in 1:2
            inverse_standardized_set[:,:,c,:] .= (training_set[:,:,c,:] .* stds[c]) .+ means[c]
        end
        return inverse_standardized_set
    end
end

function generate_or_load_data(N_dns, N_les, Re, v_train_path, c_train_path, v_test_path, c_test_path, generate_new_data, nt, dt, num_initial_conditions, num_train_conditions; dev)
    if !generate_new_data && isfile(v_train_path) && isfile(c_train_path)
        # Load existing data if available
        println("Loading existing training and test dataset...")
        v_train = deserialize(v_train_path)
        c_train = deserialize(c_train_path)
        v_test = deserialize(v_test_path)
        c_test = deserialize(c_test_path)
    else
        println("Generating new training and test dataset...")

        create_right_hand_side(setup, psolver) = function right_hand_side(u, p, t)
            u = pad_circular(u, 1; dims = 1:2)
            F = INS.momentum(u, nothing, t, setup)
            F = F[2:end-1, 2:end-1, :]
            F = pad_circular(F, 1; dims = 1:2)
            PF = INS.project(F, setup; psolver)
            PF[2:end-1, 2:end-1, :]
        end

        backend = CUDABackend();
        # Setup DNS - Grid
        x_dns = LinRange(0.0, 1.0, N_dns + 1), LinRange(0.0, 1.0, N_dns + 1);
        setup_dns = INS.Setup(; x=x_dns, Re=Re, backend);
        # Setup DNS - psolver
        psolver_dns = INS.psolver_spectral(setup_dns);
        # Setup LES - Grid 500
        x_les = LinRange(0.0, 1.0, N_les + 1), LinRange(0.0, 1.0, N_les + 1);
        setup_les = INS.Setup(; x=x_les, Re=Re, backend);
        # Setup LES - psolver
        psolver_les = INS.psolver_spectral(setup_les);
        # SciML-compatible right hand side function
        f_dns = create_right_hand_side(setup_dns, psolver_dns);
        f_les = create_right_hand_side(setup_les, psolver_les); 
    
        # Initialize empty arrays for concatenating training data
        v_train = nothing;
        c_train = nothing;
        v_test = nothing; 
        c_test = nothing; 

        anim = Animation()
        for cond in 1:num_initial_conditions
            println("Generating data for initial condition $cond")
            # GPU version
            v = zeros(N_les, N_les, 2, nt + 1, 1) |> dev;
            c = zeros(N_les, N_les, 2, nt + 1, 1) |> dev;
            global u = INS.random_field(setup_dns, 0.0) |> dev;
            global u = u[2:end-1, 2:end-1, :];
            nburn = 500; # number of steps to stabilize the simulation before collecting data.
            for i = 1:nburn
                u = step_rk4(u, dt, f_dns)
            end
            # Generate time evolution data
            global t = 0;
            for i = 1:nt+1
                # Update DNS solution at each timestep
                if i > 1
                    global u
                    t += dt
                    u = step_rk4(u, dt, f_dns)
                end
                u = pad_circular(u, 1; dims = 1:2);
                comp = div(N_dns, N_les)
                ubar = face_average_syver(u, setup_les, comp);

                ubar = ubar[2:end-1, 2:end-1, :];
                u = u[2:end-1, 2:end-1, :];
                input_filtered_RHS = pad_circular(f_dns(u, nothing, 0.0), 1; dims=1:2);
                filtered_RHS = face_average_syver(input_filtered_RHS, setup_les, comp);

                filtered_RHS = filtered_RHS[2:end-1, 2:end-1, :];      
                RHS_ubar = f_les(ubar, nothing, 0.0);
                c[:, :, :, i, 1] = Array(filtered_RHS - RHS_ubar);
                v[:, :, :, i, 1] = Array(ubar);
                # Generate visualizations every 10 steps
                # if i % 100 == 0
                #     ω_dns = Array(INS.vorticity(pad_circular(u, 1; dims = 1:2), setup_dns))
                #     ω_les = Array(INS.vorticity(pad_circular(ubar, 1; dims = 1:2), setup_les))
                #     ω_dns = ω_dns[2:end-1, 2:end-1];
                #     ω_les = ω_les[2:end-1, 2:end-1];
                #     title_dns = @sprintf("Vorticity (DNS), t = %.3f", t)
                #     title_les = @sprintf("Vorticity (Filtered DNS), t = %.3f", t)
                #     p1 = Plots.heatmap(ω_dns'; xlabel = "x", ylabel = "y", title = title_dns, color=:viridis)
                #     p2 = Plots.heatmap(ω_les'; xlabel = "x", ylabel = "y", title = title_les, color=:viridis)
                #     fig = Plots.plot(p1, p2, layout = (1, 2), size=(1200, 400))
                #     frame(anim, fig)
                # end
            end

            if cond <= num_train_conditions
                v_train = v_train === nothing ? v : cat(v_train, v; dims=5)
                c_train = c_train === nothing ? c : cat(c_train, c; dims=5)
            else
                v_test = v_test === nothing ? v : cat(v_test, v; dims=5)
                c_test = c_test === nothing ? c : cat(c_test, c; dims=5)
            end
        end
        # gif(anim, "figures/vorticity_comparison_animation.gif")

        println("Saving generated dataset...")
        serialize(v_train_path, v_train)
        serialize(c_train_path, c_train)
        serialize(v_test_path, v_test)
        serialize(c_test_path, c_test)
    end
    return v_train, c_train, v_test, c_test 
end

function generate_or_load_standardized_data(v_train_standardized_path, c_train_standardized_path, v_test_standardized_path, c_test_standardized_path, generate_new_data, v_train, c_train, v_test, c_test, state_means_path, state_std_path, closure_means_path, closure_std_path)
    if !generate_new_data && isfile(v_train_standardized_path) && isfile(c_train_standardized_path)
        # Load existing data if available
        println("Loading existing standardized training and test dataset...")
        v_train_standardized = deserialize(v_train_standardized_path)
        c_train_standardized = deserialize(c_train_standardized_path)
        v_test_standardized = deserialize(v_test_standardized_path)
        c_test_standardized = deserialize(c_test_standardized_path)

        println("Loading means and std values...")
        state_means = deserialize(state_means_path)
        state_std = deserialize(state_std_path)
        closure_means = deserialize(closure_means_path)
        closure_std = deserialize(closure_std_path)
    else
        println("Standardize the training and test dataset...")
        state_means, state_std = compute_mean_std(v_train);
        closure_means, closure_std = compute_mean_std(c_train);

        v_train_standardized = standardize_training_set_per_channel(v_train, state_means, state_std);
        v_test_standardized = standardize_training_set_per_channel(v_test, state_means, state_std);

        c_train_standardized = standardize_training_set_per_channel(c_train, closure_means, closure_std);
        c_test_standardized = standardize_training_set_per_channel(c_test, closure_means, closure_std);

        # Save generated data and means/std
        println("Saving generated standardized dataset and mean/std values...")
        serialize(v_train_standardized_path, v_train_standardized)
        serialize(c_train_standardized_path, c_train_standardized)
        serialize(v_test_standardized_path, v_test_standardized)
        serialize(c_test_standardized_path, c_test_standardized)

        serialize(state_means_path, state_means)
        serialize(state_std_path, state_std)
        serialize(closure_means_path, closure_means)
        serialize(closure_std_path, closure_std)
    end
    return v_train_standardized, c_train_standardized, v_test_standardized, c_test_standardized, state_means, state_std, closure_means, closure_std
end

# Assuming v_train_standardized, c_train_standardized, v_test_standardized, c_test_standardized are defined
function print_min_max(dataset, name)
    println("$name:")
    println("  Min: ", minimum(dataset))
    println("  Max: ", maximum(dataset))
end

function compute_velocity_magnitude(v)
    @assert size(v, 3) == 2 "Input array must have 2 fields (v_x and v_y) in the third dimension."
    
    v_x = view(v, :, :, 1, :, :)  # Extract v_x
    v_y = view(v, :, :, 2, :, :)  # Extract v_y

    # Perform element-wise operations compatible with CPU and GPU
    return sqrt.(v_x.^2 .+ v_y.^2)
end

function plot_velocity_magnitudes(v_train, v_test, c_train, c_test, v_train_standardized, v_test_standardized, c_train_standardized, c_test_standardized, time_step, trajectory)
    # Compute velocity magnitudes for all datasets
    datasets = [
        ("v_train", compute_velocity_magnitude(v_train)[:, :, time_step, trajectory]),
        ("c_train", compute_velocity_magnitude(c_train)[:, :, time_step, trajectory]),
        ("v_test", compute_velocity_magnitude(v_test)[:, :, time_step, trajectory]),
        ("c_test", compute_velocity_magnitude(c_test)[:, :, time_step, trajectory]),
        ("v_train_stand", compute_velocity_magnitude(v_train_standardized)[:, :, time_step, trajectory]),
        ("c_train_stand", compute_velocity_magnitude(c_train_standardized)[:, :, time_step, trajectory]),
        ("v_test_stand", compute_velocity_magnitude(v_test_standardized)[:, :, time_step, trajectory]),
        ("c_test_stand", compute_velocity_magnitude(c_test_standardized)[:, :, time_step, trajectory])      
    ]

    # Convert GPU arrays to CPU before plotting
    plots = []
    for (label, data) in datasets
        data_cpu = Array(data)  # Ensure the data is on the CPU
        push!(plots, Plots.heatmap(data_cpu'; xlabel = "x", ylabel = "y", title = label, color=:viridis))
    end

    # Arrange the plots in a grid
    fig = Plots.plot(plots..., layout = (2, 4), size=(2400, 800))
    savefig(fig, "figures/velocity_magnitude_datasets.png")
    println("Plot saved as figures/velocity_magnitude_datasets.png")
end

function create_training_sets(c_train, v_train)
    # Get dimensions
    x, y, num_components, num_time_steps, num_trajectories = size(c_train)

    # Initialize arrays for GPU compatibility
    initial_sample = CUDA.zeros(Float32, x, y, num_components, num_time_steps - 1, num_trajectories)
    target_sample = CUDA.zeros(Float32, x, y, num_components, num_time_steps - 1, num_trajectories)
    target_label = CUDA.zeros(Float32, x, y, num_components, num_time_steps - 1, num_trajectories)

    for i in 1:num_time_steps - 1
        # Assign slices directly into the preallocated arrays
        initial_sample[:, :, :, i, :] .= c_train[:, :, :, i, :]
        target_sample[:, :, :, i, :] .= c_train[:, :, :, i + 1, :]
        target_label[:, :, :, i, :] .= v_train[:, :, :, i + 1, :]
    end

    return initial_sample, target_sample, target_label
end