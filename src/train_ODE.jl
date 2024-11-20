using Zygote
using Statistics
using Optimisers
using BSON
using CUDA
using LuxCUDA


function initialize_or_load_model(model_name::String, is_gaussian::Bool, network::Any, load_path::Union{String, Nothing} = nothing; dev)
    if isnothing(load_path)
        # Initialize new models
        println("Initializing new models with name: $model_name")
        
        # Drift term initialization
        ps_drift, st_drift = Lux.setup(Random.default_rng(), network) |> dev
        opt_drift = Optimisers.setup(Adam(1.0e-3, (0.9f0, 0.99f0), 1e-10), ps_drift)
    else
        println("Loading models from path: $load_path")
        ps_drift, st_drift, opt_drift = load_model(load_path; dev)
    end
    return ps_drift, st_drift, opt_drift
end

# Function to load model parameters and optimizer states with structure checks
function load_model(file_path; dev)
    if dev==cpu_device()
        data = BSON.load(file_path)
        ps_drift = data[:ps_drift]
        st_drift = data[:st_drift]
        opt_drift = data[:opt_drift]

        if haskey(data, :ps_denoiser)
            ps_denoiser = data[:ps_denoiser]
            st_denoiser = data[:st_denoiser]
            opt_denoiser = data[:opt_denoiser]
            println("Loaded model and optimizer states (drift and denoiser) from $file_path")
            return ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser
        else
            println("Loaded model and optimizer states (drift) from $file_path")
            return ps_drift, st_drift, opt_drift
        end
    else
        # Load data from the BSON file
        data = BSON.load(file_path)
        # Check and print the structure and contents of each parameter
        ps_drift = data[:ps_drift_cpu]
        # Adapt ps_drift to the appropriate device
        ps_drift = deepcopy(ps_drift) |> dev

        # Repeat similar checks for st_drift and opt_drift
        st_drift = data[:st_drift_cpu]
        st_drift = deepcopy(st_drift) |> dev

        opt_drift = data[:opt_drift_cpu]
        opt_drift = deepcopy(opt_drift) |> dev
        println("Loaded model and optimizer states (drift only) from $file_path")
        return ps_drift, st_drift, opt_drift
    end
end


function save_model(file_path, ps_drift, st_drift, opt_drift; dev)
    if dev==cpu_device()
        BSON.@save file_path ps_drift st_drift opt_drift
        println("Model and optimizer states (drift) saved to $file_path")
    else
        ps_drift_cpu = ps_drift |> cpu_device()
        st_drift_cpu = st_drift |> cpu_device()
        opt_drift_cpu = opt_drift |> cpu_device()
        BSON.@save file_path ps_drift_cpu st_drift_cpu opt_drift_cpu
        println("Model and optimizer states (drift) saved to $file_path on CPU.")
    end
end

# function get_minibatch_MNIST(images, batch_size, batch_index)
#     start_index = (batch_index - 1) * batch_size + 1
#     end_index = min(batch_index * batch_size, size(images,1))
#     minibatch = images[start_index:end_index,:,:,:]
#     minibatch = permutedims(minibatch, (2,3,4,1))
#     return minibatch # Shape: (32, 32, 1, N_b) 
# end

function get_minibatch_NS(images, batch_size, batch_index)
    start_index = (batch_index - 1) * batch_size + 1
    end_index = min(batch_index * batch_size, size(images, 4))  # Adjusted for 4th dimension
    minibatch = images[:,:,:,start_index:end_index]
    return minibatch # Shape: (128, 128, 1, B)
end

function loss_fn(velocity, dI_dt_sample)
    # Compute the loss
    # loss = velocity .^ 2 .- 2 .* (velocity .* dI_dt_sample)
    loss = mean(abs2, velocity - dI_dt_sample)

    # mean_loss = mean(loss)
    return loss
end


function train!(velocity_cnn, ps, st, opt, num_epochs, batch_size, train_images, train_labels, num_batches, dev, model_name, save_path)
    init_learning_rate = 1.0e-3
    min_learning_rate = 1.0e-7
    for epoch in 1:num_epochs
        println("Epoch $epoch")
        
        # Shuffle indices for data at the start of each epoch
        shuffled_indices = randperm(size(train_images, 4))  # Assuming data is in (H, W, C, N) format

        # Shuffle the data
        train_images = train_images[:, :, :, shuffled_indices]
        train_labels = train_labels[:, :, :, shuffled_indices]

        new_learning_rate = min_learning_rate .+ 0.5f0 .* (init_learning_rate - min_learning_rate) .* (1 .+ cos.(epoch ./ num_epochs .* π))
        Optimisers.adjust!(opt, new_learning_rate)

        epoch_loss = 0.0
        for batch_index in 1:num_batches-1
            # Sample a batch from the gaussian distribution (z) and target distribution (MNIST data)
            # initial_sample = Float32.(get_minibatch(train_gaussian_images, batch_size, batch_index)) |> dev  # shape: (32, 32, 2, N_b)
            # initial_sample = Float32.(randn(32, 32, 1, batch_size)) |> dev  # shape: (32, 32, 1, N_b)
            target_sample = get_minibatch_NS(train_images, batch_size, batch_index) |> dev  # shape: (32, 32, 2, N_b)
            initial_sample = Float32.(randn(size(target_sample))) |> dev # shape: (32,32,2,N_b)
            # Sample the corresponding train_labels of the target samples
            target_labels_sample = get_minibatch_NS(train_labels, batch_size, batch_index) |> dev # shape: (32,32,1,N_b)
            # Sample time t from a uniform distribution between 0 and 1
            t_sample = Float32.(reshape(rand(Float32, batch_size), 1, 1, 1, batch_size)) |> dev  # shape: (1, 1, 1, N_b)
            # Sample the noise for the stochastic interpolant
            z_sample = Float32.(randn(size(target_sample))) |> dev   # shape: (32,32,2,N_b)

            # Define the loss function closure for gradient calculation
            loss_fn_closure = (ps_) -> begin
                # Compute the interpolant I_t and its time derivative ∂t I_t
                I_sample = stochastic_interpolant(initial_sample, target_sample, z_sample, t_sample) # shape: (32, 32, 1, N_b)
                dI_dt_sample = time_derivative_stochastic_interpolant(initial_sample, target_sample, z_sample, t_sample) # shape: (32, 32, 1, N_b)
                # Compute velocity using the neural network
                velocity, _ = Lux.apply(velocity_cnn, (I_sample, t_sample, target_labels_sample), ps_, st) # shape: (32, 32, 1, N_b)
                return loss_fn(velocity, dI_dt_sample), st
            end

            (loss, st), pb_f = Zygote.pullback(
                p -> loss_fn_closure(p), ps
            ); 

            # println("structure of pb_f: ", typeof(pb_f))
            epoch_loss += loss

            gs = pb_f((one(loss), nothing))[1];
            # println("Gradient norm: ", norm(gs))
            opt, ps = Optimisers.update!(opt, ps, gs)

        end
        epoch_loss /= num_batches
        println("Epoch loss: $epoch_loss")
    end

    # Save the model at the end of the full training process
    println("Training completed. Saving the final model.")
    save_model("$save_path/$model_name.bson", ps, st, opt; dev)

    return ps, st
end