using Zygote
using Statistics
using Optimisers
using BSON

# Function to save the model parameters and optimizer states
function save_model(file_path, ps_drift, st_drift, opt_drift, ps_denoiser=nothing, st_denoiser=nothing, opt_denoiser=nothing)
    if ps_denoiser !== nothing && st_denoiser !== nothing && opt_denoiser !== nothing
        BSON.@save file_path ps_drift st_drift opt_drift ps_denoiser st_denoiser opt_denoiser
        println("Model and optimizer states (drift and denoiser) saved to $file_path")
    else
        BSON.@save file_path ps_drift st_drift opt_drift
        println("Model and optimizer states (drift) saved to $file_path")
    end
end

# Function to load the model parameters and optimizer states
function load_model(file_path)
    data = BSON.load(file_path)
    ps_drift = data[:ps_drift] |> gpu_device()
    st_drift = data[:st_drift] |> gpu_device()
    opt_drift = data[:opt_drift]

    if haskey(data, :ps_denoiser)
        ps_denoiser = data[:ps_denoiser] |> gpu_device()
        st_denoiser = data[:st_denoiser] |> gpu_device()
        opt_denoiser = data[:opt_denoiser] 
        println("Loaded model and optimizer states (drift and denoiser) from $file_path")
        return ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser
    else
        println("Loaded model and optimizer states (drift) from $file_path")
        return ps_drift, st_drift, opt_drift
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
    # loss = sum((velocity - dI_dt_sample) .^ 2) # For real numbers
    loss = mean(abs2, velocity - dI_dt_sample) # Loss function for imaginary numbers.
    # Changed sum to mean --> smaller loss values. 

    # # Check for NaN or Inf in the loss using broadcasting
    # if any(isnan.(loss)) || any(isinf.(loss))
    #     println("Loss contains NaN or Inf")
    # end

    # mean_loss = mean(loss)
    return loss
end

function train!(velocity_cnn, ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser, num_epochs, batch_size, train_gaussian_images, train_images, train_labels, num_batches, dev, is_gaussian, save_path)
    for epoch in 1:num_epochs
        println("Epoch $epoch")

        epoch_drift_loss = 0.0
        if !is_gaussian
            epoch_denoiser_loss = 0.0
        end
        for batch_index in 1:num_batches-1
            # Sample a batch from the gaussian distribution (z) and target distribution (MNIST data)
            initial_sample = get_minibatch_NS(train_gaussian_images, batch_size, batch_index) |> dev  # shape: (32, 32, 1, N_b)
            # initial_sample = Float32.(randn(32, 32, 1, batch_size)) |> dev  # shape: (32, 32, 1, N_b)
            target_sample = get_minibatch_NS(train_images, batch_size, batch_index) |> dev  # shape: (32, 32, 1, N_b)
            # Sample the corresponding train_labels of the target samples
            target_labels_sample = get_minibatch_NS(train_labels, batch_size, batch_index) |> dev # shape: (32,32,1,N_b)
            # Sample time t from a uniform distribution between 0 and 1
            t_sample = reshape(rand(Float32, batch_size), 1, 1, 1, batch_size) |> dev  # shape: (1, 1, 1, N_b)
            # Sample the noise for the stochastic interpolant
            z_sample = Float32.(randn(size(initial_sample))) |> dev   # shape: (32,32,1,N_b)
            
            ## Update the weights for the drift term ##
            # Define the loss function closure for gradient calculation
            loss_drift_closure = (ps_) -> begin
                # Compute the interpolant I_t and its time derivative ∂t I_t
                I_sample = stochastic_interpolant(initial_sample, target_sample, z_sample, t_sample) # shape: (32, 32, 1, N_b)
                dI_dt_sample = time_derivative_stochastic_interpolant(initial_sample, target_sample, z_sample, t_sample) # shape: (32, 32, 1, N_b)
                # Compute velocity using the neural network
                velocity, _ = Lux.apply(velocity_cnn, (I_sample, t_sample, target_labels_sample), ps_, st_drift) # shape: (32, 32, 1, N_b)
                return loss_fn(velocity, dI_dt_sample), st_drift
            end
        
            (loss_drift, st_drift), pb_drift_f = Zygote.pullback(
                p -> loss_drift_closure(p), ps_drift
            ); 

            # println("structure of pb_f: ", typeof(pb_f))
            epoch_drift_loss += loss_drift

            gs_drift = pb_drift_f((one(loss_drift), nothing))[1];
            # println("Gradient norm: ", norm(gs))
            opt_drift, ps_drift = Optimisers.update!(opt_drift, ps_drift, gs_drift)  
            
            ## Update the weights for the denoiser term ##
            if !is_gaussian
                # Define the loss function closure for gradient calculation
                loss_denoiser_closure = (ps_) -> begin
                    # Compute the interpolant I_t and its time derivative ∂t I_t
                    I_sample = stochastic_interpolant(initial_sample, target_sample, z_sample, t_sample) # shape: (32, 32, 1, N_b)
                    # Compute velocity using the neural network
                    denoiser, _ = Lux.apply(velocity_cnn, (I_sample, t_sample, target_labels_sample), ps_, st_denoiser) # shape: (32, 32, 1, N_b)
                    return loss_fn(denoiser, z_sample), st_denoiser
                end
    
                (loss_denoiser, st_denoiser), pb_denoiser_f = Zygote.pullback(
                    p -> loss_denoiser_closure(p), ps_denoiser
                ); 

                # println("structure of pb_f: ", typeof(pb_f))
                epoch_denoiser_loss += loss_denoiser

                gs_denoiser = pb_denoiser_f((one(loss_denoiser), nothing))[1];
                # println("Gradient norm: ", norm(gs))
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
        save_model("$save_path/final_model.bson", ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser)
    else
        save_model("$save_path/final_model.bson", ps_drift, st_drift, opt_drift)

    end

    # Save the model at the end of the full training process
    println("Training completed. Saving the final model.")
    if !is_gaussian
        save_model("$save_path/final_model.bson", ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser)
    else
        save_model("$save_path/final_model.bson", ps_drift, st_drift, opt_drift)
    end


    return ps_drift, st_drift, ps_denoiser, st_denoiser
    if !is_gaussian
        return ps_denoiser, st_denoiser
    end

end



