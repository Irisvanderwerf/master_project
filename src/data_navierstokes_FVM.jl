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
    means = [mean(training_set[:,:,c,:]) for c in 1:2]
    stds = [std(training_set[:,:,c,:]) for c in 1:2]
    return means, stds
end

# Function to standardize the training set per channel
function standardize_training_set_per_channel(training_set, means, stds)
    standardized_set = similar(training_set)
    for c in 1:2
        standardized_set[:,:,c,:] .= (training_set[:,:,c,:] .- means[c]) ./ stds[c]
    end
    return standardized_set
end

# Function of the inverse standardization per channel
function inverse_standardize_set_per_channel(training_set, means, stds)
    inverse_standardized_set = similar(training_set)
    for c in 1:2
        inverse_standardized_set[:,:,c,:] .= (training_set[:,:,c,:] .* stds[c]) .+ means[c]
    end
    return inverse_standardized_set
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
        v_train = Array{Float32}[];
        c_train = Array{Float32}[];
        v_test = Array{Float32}[];
        c_test = Array{Float32}[];

        anim = Animation()
        for cond = 1:num_initial_conditions
            println("Generating data for initial condition $cond")
            # GPU version
            v = zeros(N_les, N_les, 2, nt + 1) |> dev;
            c = zeros(N_les, N_les, 2, nt + 1) |> dev;
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
                c[:, :, :, i] = Array(filtered_RHS - RHS_ubar);
                v[:, :, :, i] = Array(ubar);
                # Generate visualizations every 10 steps
                if i % 10 == 0
                    ω_dns = Array(INS.vorticity(pad_circular(u, 1; dims = 1:2), setup_dns))
                    ω_les = Array(INS.vorticity(pad_circular(ubar, 1; dims = 1:2), setup_les))
                    ω_dns = ω_dns[2:end-1, 2:end-1];
                    ω_les = ω_les[2:end-1, 2:end-1];
                    title_dns = @sprintf("Vorticity (DNS), t = %.3f", t)
                    title_les = @sprintf("Vorticity (Filtered DNS), t = %.3f", t)
                    p1 = Plots.heatmap(ω_dns'; xlabel = "x", ylabel = "y", title = title_dns, color=:viridis)
                    p2 = Plots.heatmap(ω_les'; xlabel = "x", ylabel = "y", title = title_les, color=:viridis)
                    fig = Plots.plot(p1, p2, layout = (1, 2), size=(1200, 400))
                    frame(anim, fig)
                end
            end
            if cond <= num_train_conditions
                if cond == 1 && size(v_train, 1) == 0
                    # Initialize the array on the first condition
                    v_train = Array(v)
                    c_train = Array(c)
                else
                    # Concatenate subsequent conditions
                    v_train = cat(v_train, Array(v); dims=4)
                    c_train = cat(c_train, Array(c); dims=4)
                end
            else
                if cond == num_train_conditions + 1 && size(v_test, 1) == 0
                    # Initialize the test arrays for the first test condition
                    v_test = Array(v)
                    c_test = Array(c)
                else
                    # Concatenate subsequent conditions
                    v_test = cat(v_test, Array(v); dims=4)
                    c_test = cat(c_test, Array(c); dims=4)
                end
            end
        end
        gif(anim, "vorticity_comparison_animation.gif")

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
        c_train_standardized = standardize_training_set_per_channel(c_train, closure_means, closure_std);

        v_test_standardized = standardize_training_set_per_channel(v_test, state_means, state_std);
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
