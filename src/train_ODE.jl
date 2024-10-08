using Zygote
using Statistics
using Optimisers


function get_minibatch(images, batch_size, batch_index)
    start_index = (batch_index - 1) * batch_size + 1
    end_index = min(batch_index * batch_size, size(images,1))
    minibatch = images[start_index:end_index,:,:,:]
    minibatch = permutedims(minibatch, (2,3,4,1))
    return minibatch # Shape: (32, 32, 1, N_b) 
end

function loss_fn(velocity, dI_dt_sample)
    # Compute the loss
    loss = velocity .^ 2 .- 2 .* (velocity .* dI_dt_sample)

    # Check for NaN or Inf in the loss using broadcasting
    if any(isnan.(loss)) || any(isinf.(loss))
        println("Loss contains NaN or Inf")
    end

    mean_loss = mean(loss)
    return mean_loss
end

function train!(velocity_cnn, ps, st, opt, num_epochs, batch_size, train_gaussian_images, train_images, train_labels, num_batches, dev)
    for epoch in 1:num_epochs
        println("Epoch $epoch")
        
        # shuffle the gaussian images and train_images such that you have different pairs du    ring training.
        # shuffle_ids = shuffle(1:size(train_gaussian_images, 1))
        # train_gaussian_images = train_gaussian_images[shuffle_ids, :, :]

        # shuffle_ids = shuffle(1:size(train_images, 1))
        # train_images = train_gaussian_images[shuffle_ids, :, :]


        epoch_loss = 0.0
        for batch_index in 1:num_batches-1
            # Sample a batch from the gaussian distribution (z) and target distribution (MNIST data)
            initial_sample = Float32.(get_minibatch(train_gaussian_images, batch_size, batch_index)) |> dev  # shape: (32, 32, 1, N_b)
            # initial_sample = Float32.(randn(32, 32, 1, batch_size)) |> dev  # shape: (32, 32, 1, N_b)
            target_sample = Float32.(get_minibatch(train_images, batch_size, batch_index)) |> dev  # shape: (32, 32, 1, N_b)
            # Sample the corresponding train_labels of the target samples
            target_labels_sample = Float32.(get_minibatch(train_labels, batch_size, batch_index)) |> dev # shape: (32,32,1,N_b)
            # Sample time t from a uniform distribution between 0 and 1
            t_sample = Float32.(reshape(rand(Float32, batch_size), 1, 1, 1, batch_size)) |> dev  # shape: (1, 1, 1, N_b)
            # Sample the noise for the stochastic interpolant
            z_sample = randn(Float32, 32, 32, 1, batch_size) |> dev   # shape: (32,32,1,N_b)

            # Define the loss function closure for gradient calculation
            loss_fn_closure = (ps_) -> begin
                # Compute the interpolant I_t and its time derivative ∂t I_t
                I_sample = Float32.(stochastic_interpolant(initial_sample, target_sample, z_sample, t_sample)) # shape: (32, 32, 1, N_b)
                dI_dt_sample = Float32.(time_derivative_stochastic_interpolant(initial_sample, target_sample, z_sample, t_sample)) # shape: (32, 32, 1, N_b)
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
    return ps, st
end