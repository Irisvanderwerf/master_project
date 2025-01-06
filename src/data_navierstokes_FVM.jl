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

using BSON

z = CUDA.functional() ? CUDA.zeros : (s...) -> zeros(Float32, s...)
ArrayType = CUDA.functional() ? CuArray : Array
CUDA.allowscalar(false)

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

function compute_mean_std(training_set)
    means = [mean(training_set[:,:,c,:,:]) for c in 1:2]
    stds = [std(training_set[:,:,c,:,:]) for c in 1:2]
    return means, stds
end

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

function save_large_bson(filepath, data)
    Base.@eval BSON begin
        function bson_primitive(io::IO, x::Int64)
            write(io, x)
        end
    end
    max_chunk_size = 10^7 
    dims = size(data)
    element_size = sizeof(eltype(data))
    chunk_size = max(1, div(max_chunk_size, dims[1] * dims[2] * dims[3] * element_size))

    metadata_filepath = filepath * "_metadata.bson" 
    BSON.@save(metadata_filepath, dims=dims)

    existing_files = filter(x -> occursin("$(basename(filepath))_part_", x), readdir(dirname(filepath)))
    foreach(f -> rm(joinpath(dirname(filepath), f)), existing_files)

    for i in 1:chunk_size:dims[4]
        part = data[:, :, :, i:min(i + chunk_size - 1, dims[4])]
        part_filepath = joinpath(dirname(filepath), "$(basename(filepath))_part_$i.bson")
        BSON.@save(part_filepath, data=part)
    end
end

function load_large_bson(filepath)
    metadata_filepath = filepath * "_metadata.bson"

    if isfile(metadata_filepath)
        metadata = BSON.load(metadata_filepath)
    else
        error("Metadata file not found for $(filepath)")
    end

    parts = []
    part_files = sort(filter(x -> occursin("$(basename(filepath))_part_", x), readdir(dirname(filepath))))
    for file in part_files
        part_data = BSON.load(joinpath(dirname(filepath), file))[:data]
        push!(parts, part_data)
    end

    return cat(parts..., dims=4)
end

function generate_or_load_data(N_dns, LES_resolutions, Re, output_dir, generate_new_data, nt, dt, num_trajectories; dev)
    isdir(output_dir) || mkdir(output_dir)
    data_v = Dict{Int, Array}()
    data_c = Dict{Int, Array}()

    if !generate_new_data
        println("Loading existing trajectories...")
        for N_les in LES_resolutions
            resolution_dir = joinpath(output_dir, "LES_$(N_les)")
            v_all = []
            c_all = []
        
            for cond in 1:num_trajectories
                filepath_v = joinpath(resolution_dir, "v_cond_$cond")
                filepath_c = joinpath(resolution_dir, "c_cond_$cond")
        
                part_files_v = filter(x -> startswith(x, "v_cond_$(cond)_part_"), readdir(resolution_dir))
                part_files_c = filter(x -> startswith(x, "c_cond_$(cond)_part_"), readdir(resolution_dir))
        
                if isempty(part_files_v) || isempty(part_files_c)
                    error("Missing trajectory files for LES resolution $N_les condition $cond in $resolution_dir")
                end
        
                v = load_large_bson(filepath_v)
                c = load_large_bson(filepath_c)
        
                push!(v_all, v)
                push!(c_all, c)
            end
        
            data_v[N_les] = cat(v_all..., dims=5)
            data_c[N_les] = cat(c_all..., dims=5)
        end       
    else
        println("Generating new dataset for each LES resolution...")

        create_right_hand_side(setup, psolver) = function right_hand_side(u, p, t)
            u = pad_circular(u, 1; dims = 1:2)
            F = INS.momentum(u, nothing, t, setup)
            F = F[2:end-1, 2:end-1, :]
            F = pad_circular(F, 1; dims = 1:2)
            PF = INS.project(F, setup; psolver)
            PF[2:end-1, 2:end-1, :]
        end

        backend = CUDABackend()
        x_dns = LinRange(0.0, 1.0, N_dns + 1), LinRange(0.0, 1.0, N_dns + 1)
        setup_dns = INS.Setup(; x=x_dns, Re=Re, backend)
        psolver_dns = INS.psolver_spectral(setup_dns)
        f_dns = create_right_hand_side(setup_dns, psolver_dns)

        for N_les in LES_resolutions
            println("Generating data for LES resolution $N_les...")
            resolution_dir = joinpath(output_dir, "LES_$(N_les)")
            isdir(resolution_dir) || mkdir(resolution_dir)

            x_les = LinRange(0.0, 1.0, N_les + 1), LinRange(0.0, 1.0, N_les + 1)
            setup_les = INS.Setup(; x=x_les, Re=Re, backend)
            psolver_les = INS.psolver_spectral(setup_les)
            f_les = create_right_hand_side(setup_les, psolver_les)

            v_all = []
            c_all = []

            for cond in 1:num_trajectories
                println("Generating data for trajectory $cond at LES resolution $N_les")
                v = []
                c = []

                global u = INS.random_field(setup_dns, 0.0) |> dev
                global u = u[2:end-1, 2:end-1, :]

                nburn = 5000
                for i = 1:nburn
                    u = step_rk4(u, dt, f_dns)
                end

                global t = 0
                for i = 1:nt+1
                    if i > 1
                        global u
                        t += dt
                        u = step_rk4(u, dt, f_dns)
                    end

                    if i % 10 == 0
                        u = pad_circular(u, 1; dims = 1:2)
                        comp = div(N_dns, N_les)
                        ubar = face_average_syver(u, setup_les, comp)

                        ubar = ubar[2:end-1, 2:end-1, :]
                        u = u[2:end-1, 2:end-1, :]
                        input_filtered_RHS = pad_circular(f_dns(u, nothing, 0.0), 1; dims=1:2)
                        filtered_RHS = face_average_syver(input_filtered_RHS, setup_les, comp)

                        filtered_RHS = filtered_RHS[2:end-1, 2:end-1, :]
                        RHS_ubar = f_les(ubar, nothing, 0.0)

                        push!(c, Array(filtered_RHS - RHS_ubar))
                        push!(v, Array(ubar))
                    end
                end

                v = permutedims(cat(v..., dims=4), (1, 2, 3, 4))
                c = permutedims(cat(c..., dims=4), (1, 2, 3, 4))

                filepath_v = joinpath(resolution_dir, "v_cond_$cond")
                filepath_c = joinpath(resolution_dir, "c_cond_$cond")

                save_large_bson(filepath_v, v)
                save_large_bson(filepath_c, c)

                push!(v_all, v)
                push!(c_all, c)
            end

            data_v[N_les] = cat(v_all..., dims=5)
            data_c[N_les] = cat(c_all..., dims=5)
        end
    end

    return data_v, data_c
end

function generate_or_load_stand_data(
    raw_data_v,
    raw_data_c,
    standardized_dir::String,
    generate_new_data::Bool
)
    isdir(standardized_dir) || mkdir(standardized_dir)
    standardized_data_v = Dict{Int, Array}()
    standardized_data_c = Dict{Int, Array}()
    stats = Dict{Int, Dict{Symbol, Array}}()
    for (resolution, v_data) in raw_data_v
        println("Processing LES resolution $resolution...")
        resolution_dir = joinpath(standardized_dir, "LES_$resolution")
        isdir(resolution_dir) || mkdir(resolution_dir)
        state_means_path = joinpath(resolution_dir, "state_means.bson")
        state_std_path = joinpath(resolution_dir, "state_std.bson")
        closure_means_path = joinpath(resolution_dir, "closure_means.bson")
        closure_std_path = joinpath(resolution_dir, "closure_std.bson")
        num_trajectories = size(v_data, 5)
        if !generate_new_data && isfile(state_means_path) && isfile(state_std_path) &&
           isfile(closure_means_path) && isfile(closure_std_path)
            println("Loading existing statistics for resolution $resolution...")

            stats[resolution] = Dict(
                :state_means => deserialize(state_means_path),
                :state_std => deserialize(state_std_path),
                :closure_means => deserialize(closure_means_path),
                :closure_std => deserialize(closure_std_path)
            )

            v_all = []
            c_all = []

            for cond in 1:num_trajectories
                filepath_v = joinpath(resolution_dir, "v_cond_$cond")
                filepath_c = joinpath(resolution_dir, "c_cond_$cond")

                part_files_v = filter(x -> startswith(x, "v_cond_$(cond)_part_"), readdir(resolution_dir))
                part_files_c = filter(x -> startswith(x, "c_cond_$(cond)_part_"), readdir(resolution_dir))

                if isempty(part_files_v) || isempty(part_files_c)
                    error("Missing trajectory files for LES resolution $N_les condition $cond in $resolution_dir")
                end

                println("Loading standardized trajectory $cond for resolution $resolution from parts...")
                v = load_large_bson(filepath_v)
                c = load_large_bson(filepath_c)
                push!(v_all, v)
                push!(c_all, c)
            end

            standardized_data_v[resolution] = cat(v_all..., dims=5)
            standardized_data_c[resolution] = cat(c_all..., dims=5)
        else
            println("Standardizing data for resolution $resolution...")
            state_means, state_std = compute_mean_std(v_data)
            closure_means, closure_std = compute_mean_std(raw_data_c[resolution])

            println("Saving statistics for resolution $resolution...")
            serialize(state_means_path, state_means)
            serialize(state_std_path, state_std)
            serialize(closure_means_path, closure_means)
            serialize(closure_std_path, closure_std)

            stats[resolution] = Dict(
                :state_means => state_means,
                :state_std => state_std,
                :closure_means => closure_means,
                :closure_std => closure_std
            )

            v_all = []
            c_all = []

            for cond in 1:num_trajectories
                println("Standardizing trajectory $cond for resolution $resolution...")

                v_traj = v_data[:, :, :, :, cond]
                c_traj = raw_data_c[resolution][:, :, :, :, cond]

                v_standardized = standardize_training_set_per_channel(
                    v_traj,
                    state_means,
                    state_std
                )
                c_standardized = standardize_training_set_per_channel(
                    c_traj,
                    closure_means,
                    closure_std
                )

                filepath_v = joinpath(resolution_dir, "v_cond_$cond")
                filepath_c = joinpath(resolution_dir, "c_cond_$cond")

                save_large_bson(filepath_v, v_standardized)
                save_large_bson(filepath_c, c_standardized)

                push!(v_all, v_standardized)
                push!(c_all, c_standardized)
            end
            standardized_data_v[resolution] = cat(v_all..., dims=5)
            standardized_data_c[resolution] = cat(c_all..., dims=5)
        end
    end

    return standardized_data_v, standardized_data_c, stats
end


function compute_velocity_magnitude(v)
    @assert size(v, 3) == 2 "Input array must have 2 fields (v_x and v_y) in the third dimension."
    v_x = view(v, :, :, 1, :, :)
    v_y = view(v, :, :, 2, :, :) 
    return sqrt.(v_x.^2 .+ v_y.^2)
end

function plot_velocity_magnitudes(v_train, c_train, v_train_standardized, c_train_standardized, time_step, trajectory, N_les)
    datasets = [
        ("v_train", compute_velocity_magnitude(v_train[N_les])[:, :, time_step, trajectory]),
        ("c_train", compute_velocity_magnitude(c_train[N_les])[:, :, time_step, trajectory]),
        ("v_train_stand", compute_velocity_magnitude(v_train_standardized[N_les])[:, :, time_step, trajectory]),
        ("c_train_stand", compute_velocity_magnitude(c_train_standardized[N_les])[:, :, time_step, trajectory]),
    ]
    plots = []
    for (label, data) in datasets
        data_cpu = Array(data)
        push!(plots, Plots.heatmap(data_cpu'; xlabel = "x", ylabel = "y", title = label, color=:viridis))
    end
    fig = Plots.plot(plots..., layout = (1, 4), size=(2400, 400))
    savefig(fig, "figures/velocity_magnitude_datasets.png")
    println("Plot saved as figures/velocity_magnitude_datasets.png")
end

function create_training_sets(c_train, v_train)
    x, y, num_components, num_time_steps, num_trajectories = size(c_train)
    initial_sample = CUDA.zeros(Float32, x, y, num_components, num_time_steps - 1, num_trajectories)
    target_sample = CUDA.zeros(Float32, x, y, num_components, num_time_steps - 1, num_trajectories)
    target_label_closure = CUDA.zeros(Float32, x, y, num_components, num_time_steps - 1, num_trajectories)
    target_label_state = CUDA.zeros(Float32, x, y, num_components, num_time_steps-1, num_trajectories)

    for i in 1:num_time_steps - 1
        initial_sample[:, :, :, i, :] .= c_train[:, :, :, i, :]
        target_sample[:, :, :, i, :] .= c_train[:, :, :, i + 1, :]
        target_label_closure[:, :, :, i, :] .= c_train[:, :, :, i, :]
        target_label_state[:, :, :, i, :] .= v_train[:, :, :, i, :]
    end

    return initial_sample, target_sample, target_label_closure, target_label_state
end