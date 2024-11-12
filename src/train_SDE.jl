using Zygote
using Statistics
using Optimisers
using BSON
using CUDA
using LuxCUDA

function initialize_or_load_model(model_name::String, is_gaussian::Bool, load_path::Union{String, Nothing} = nothing)
    if isnothing(load_path)
        # Initialize new models
        println("Initializing new models with name: $model_name")
        
        # Drift term initialization
        ps_drift, st_drift = Lux.setup(Random.default_rng(), velocity_cnn) |> dev
        opt_drift = Optimisers.setup(Adam(1.0e-3, (0.9f0, 0.99f0), 1e-10), ps_drift)

        if !is_gaussian
            # Denoiser term initialization
            ps_denoiser, st_denoiser = Lux.setup(Random.default_rng(), velocity_cnn) |> dev
            opt_denoiser = Optimisers.setup(Adam(1.0e-3, (0.9f0, 0.99f0), 1e-10), ps_denoiser)
        else
            ps_denoiser, st_denoiser, opt_denoiser = nothing, nothing, nothing
        end
    else
        # Load existing models
        println("Loading models from path: $load_path")
        ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser = load_model(load_path, is_gaussian)
    end

    return ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser, model_name
end


# Function to load model parameters and optimizer states with structure checks
function load_model(file_path, is_gaussian)
    # Load data from the BSON file
    data = BSON.load(file_path)

    # Check and print the structure and contents of each parameter
    ps_drift = data[:ps_drift_cpu]
    # Adapt ps_drift to the appropriate device
    ps_drift = deepcopy(ps_drift) |> gpu_device()

    # Repeat similar checks for st_drift and opt_drift
    st_drift = data[:st_drift_cpu]
    st_drift = deepcopy(st_drift) |> gpu_device()

    opt_drift = data[:opt_drift_cpu]
    opt_drift = deepcopy(opt_drift) |> gpu_device()

    # Check for optional denoiser parameters
    if !is_gaussian   
        # ps_denoiser
        ps_denoiser = data[:ps_denoiser_cpu]
        ps_denoiser = deepcopy(ps_denoiser) |> gpu_device()

        # st_denoiser
        st_denoiser = data[:st_denoiser_cpu]
        st_denoiser = deepcopy(st_denoiser) |> gpu_device()

        # opt_denoiser
        opt_denoiser = data[:opt_denoiser_cpu]
        opt_denoiser = deepcopy(opt_denoiser) |> gpu_device()

        println("Loaded model and optimizer states (drift and denoiser) from $file_path")
        return ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser
    else
        println("Loaded model and optimizer states (drift only) from $file_path")
        return ps_drift, st_drift, opt_drift
    end
end

function save_model(file_path, ps_drift, st_drift, opt_drift, ps_denoiser=nothing, st_denoiser=nothing, opt_denoiser=nothing)
    # Move all parameters to the CPU
    ps_drift_cpu = ps_drift |> cpu_device()
    st_drift_cpu = st_drift |> cpu_device()
    opt_drift_cpu = opt_drift |> cpu_device() # Optimizers usually store small data that doesn’t need adaptation

    # Check if denoiser parameters are provided
    if ps_denoiser !== nothing && st_denoiser !== nothing && opt_denoiser !== nothing
        ps_denoiser_cpu = ps_denoiser |> cpu_device()
        st_denoiser_cpu = st_denoiser |> cpu_device()
        opt_denoiser_cpu = opt_denoiser |> cpu_device()
        BSON.@save file_path ps_drift_cpu st_drift_cpu opt_drift_cpu ps_denoiser_cpu st_denoiser_cpu opt_denoiser_cpu
        println("Model and optimizer states (drift and denoiser) saved to $file_path on CPU.")
    else
        BSON.@save file_path ps_drift_cpu st_drift_cpu opt_drift_cpu
        println("Model and optimizer states (drift) saved to $file_path on CPU.")
    end
end

function get_minibatch_NS(images, batch_size, batch_index)
    start_index = (batch_index - 1) * batch_size + 1
    end_index = min(batch_index * batch_size, size(images, 4))  # Adjusted for 4th dimension
    minibatch = images[:,:,:,start_index:end_index]
    return minibatch # Shape: (128, 128, 1, B)
end

function loss_fn(velocity, dI_dt_sample)
    # Compute the loss
    # loss = velocity .^ 2 .- 2 .* (velocity .* dI_dt_sample)
    # loss = mean((velocity - dI_dt_sample) .^ 2) # For real numbers
    loss = mean(abs2, velocity - dI_dt_sample) # Loss function for imaginary numbers.

    # mean_loss = mean(loss)
    return loss
end

function train!(velocity_cnn, ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser, num_epochs, batch_size, train_gaussian_images, train_images, train_labels, num_batches, dev, is_gaussian, model_name, save_path)
    init_learning_rate = 1.0e-3
    min_learning_rate = 1.0e-6

    for epoch in 1:num_epochs
        println("Epoch $epoch")

        # Shuffle indices for data at the start of each epoch
        shuffled_indices = randperm(size(train_images, 4))  # Assuming data is in (H, W, C, N) format

        # Shuffle the data
        train_gaussian_images = train_gaussian_images[:, :, :, shuffled_indices]
        train_images = train_images[:, :, :, shuffled_indices]
        train_labels = train_labels[:, :, :, shuffled_indices]

        # Learning rate scheduler
        new_learning_rate = min_learning_rate .+ 0.5f0 .* (init_learning_rate - min_learning_rate) .* (1 .+ cos.(epoch ./ num_epochs .* π))
        Optimisers.adjust!(opt_drift, new_learning_rate)
        if !is_gaussian
            Optimisers.adjust!(opt_denoiser, new_learning_rate)
        end

        epoch_drift_loss = 0.0
        epoch_denoiser_loss = 0.0

        for batch_index in 1:num_batches-1
            initial_sample = get_minibatch_NS(train_gaussian_images, batch_size, batch_index) |> dev
            target_sample = get_minibatch_NS(train_images, batch_size, batch_index) |> dev
            target_labels_sample = get_minibatch_NS(train_labels, batch_size, batch_index) |> dev
            t_sample = reshape(rand(Float32, batch_size), 1, 1, 1, batch_size) |> dev
            z_sample = Float32.(randn(size(initial_sample))) |> dev

            ## Update the weights for the drift term ##
            loss_drift_closure = (ps_) -> begin
                I_sample = stochastic_interpolant(initial_sample, target_sample, z_sample, t_sample)
                dI_dt_sample = time_derivative_stochastic_interpolant(initial_sample, target_sample, z_sample, t_sample)
                velocity, _ = Lux.apply(velocity_cnn, (I_sample, t_sample, target_labels_sample), ps_, st_drift)
                return loss_fn(velocity, dI_dt_sample), st_drift
            end

            (loss_drift, st_drift), pb_drift_f = Zygote.pullback(p -> loss_drift_closure(p), ps_drift)
            epoch_drift_loss += loss_drift
            gs_drift = pb_drift_f((one(loss_drift), nothing))[1]
            opt_drift, ps_drift = Optimisers.update!(opt_drift, ps_drift, gs_drift)

            ## Update the weights for the denoiser term ##
            if !is_gaussian
                loss_denoiser_closure = (ps_) -> begin
                    I_sample = stochastic_interpolant(initial_sample, target_sample, z_sample, t_sample)
                    denoiser, _ = Lux.apply(velocity_cnn, (I_sample, t_sample, target_labels_sample), ps_, st_denoiser)
                    return loss_fn(denoiser, z_sample), st_denoiser
                end

                (loss_denoiser, st_denoiser), pb_denoiser_f = Zygote.pullback(p -> loss_denoiser_closure(p), ps_denoiser)
                epoch_denoiser_loss += loss_denoiser
                gs_denoiser = pb_denoiser_f((one(loss_denoiser), nothing))[1]
                opt_denoiser, ps_denoiser = Optimisers.update!(opt_denoiser, ps_denoiser, gs_denoiser)
            end
        end

        epoch_drift_loss /= num_batches
        println("Epoch loss of the drift term: $epoch_drift_loss")

        if !is_gaussian
            epoch_denoiser_loss /= num_batches
            println("Epoch loss of the denoiser term: $epoch_denoiser_loss")
        end
    end

    # Save the model at the end of the full training process
    println("Training completed. Saving the final model.")
    if !is_gaussian
        save_model("$save_path/$model_name.bson", ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser)
    else
        save_model("$save_path/$model_name.bson", ps_drift, st_drift, opt_drift)
    end

    return ps_drift, st_drift, ps_denoiser, st_denoiser
end

function compute_mean_error(velocity_cnn, ps_drift, st_drift, ps_denoiser, st_denoiser, 
    train_gaussian_images, train_images, train_labels, dev, is_gaussian)
    total_error = 0.0
    num_samples = size(train_images, 4)  # Assuming (H, W, C, N) format

    for i in 1:num_samples
        # Extract a single data point from the training set
        initial_sample = train_gaussian_images[:, :, :, i:i] |> dev
        target_sample = train_images[:, :, :, i:i] |> dev
        target_label = train_labels[:, :, :, i:i] |> dev

        # Create t_sample with the desired shape and a random value for each batch
        t_sample = rand(Float32, 1, 1, 1, 1) |> dev  # shape: (1, 1, 1, 1)
        z_sample = Float32.(randn(size(initial_sample))) |> dev

        # Compute the interpolant I_t for the given data point
        I_sample = stochastic_interpolant(initial_sample, target_sample, z_sample, t_sample)

        if is_gaussian
            # Compute the prediction using only the drift network
            prediction, _ = Lux.apply(velocity_cnn, (I_sample, t_sample, target_label), ps_drift, st_drift)
            real_value = time_derivative_stochastic_interpolant(initial_sample, target_sample, z_sample, t_sample)
        else
            # Compute the prediction using the denoiser network
            prediction, _ = Lux.apply(velocity_cnn, (I_sample, t_sample, target_label), ps_denoiser, st_denoiser)
            real_value = z_sample
        end

        # Compute the error for this data point and accumulate it
        data_point_error = mean(abs.(prediction .- real_value))
        total_error += data_point_error
    end

    # Compute the mean error over all data points
    mean_error = total_error / num_samples
    println("Mean error over the training set: $mean_error")

    return mean_error
end




