using Zygote
using Statistics
using Optimisers


function get_minibatch(images, batch_size, batch_index)
    start_index = (batch_index - 1) * batch_size + 1
    end_index = min(batch_index * batch_size, size(images,1))
    minibatch = images[start_index:end_index,:,:,:]
    minibatch = permutedims(minibatch, (2,3,4,1))
    return minibatch # Shape: (28 , 28, 1, batch_size) 
end

function loss_fn(velocity, dI_dt_sample)
    # println("Inside loss_fn:")
    # println("Velocity shape: ", size(velocity))
    # println("dI_dt_sample shape: ", size(dI_dt_sample))
    # Compute the loss
    loss = velocity .^ 2 - 2 .* (dI_dt_sample .* velocity)

    # Check for NaN or Inf in the loss using broadcasting
    if any(isnan.(loss)) || any(isinf.(loss))
        println("Loss contains NaN or Inf")
    end

    mean_loss = mean(loss)
    return mean_loss
end

function train!(velocity_cnn, ps, st, opt, num_epochs, batch_size, train_gaussian_images, train_images, num_batches)
    for epoch in 1:num_epochs
        println("Epoch $epoch")
        epoch_loss = 0.0
        for batch_index in 1:num_batches-1
            # Sample a batch from the gaussian distribution (z) and target distribution (MNIST data)
            z_sample = Float32.(get_minibatch(train_gaussian_images, batch_size, batch_index))  # shape: (28, 28, 1, N_b)
            target_sample = Float32.(get_minibatch(train_images, batch_size, batch_index))  # shape: (28, 28, 1, N_b)
            # Sample time t from a uniform distribution between 0 and 1
            t_sample = Float32.(reshape(rand(Float32, batch_size), 1, 1, 1, batch_size))  # shape: (1, 1, 1, N_b)

            # Define the loss function closure for gradient calculation
            loss_fn_closure = (ps_) -> begin
                # Compute the interpolant I_t and its time derivative ∂t I_t
                I_sample = Float32.(stochastic_interpolant(z_sample, target_sample, t_sample)) # shape: (28, 28, 1, N_b)
                dI_dt_sample = Float32.(time_derivative_stochastic_interpolant(z_sample, target_sample, t_sample)) # shape: (28, 28, 1, N_b)
                # Compute velocity using the neural network
                velocity, _ = Lux.apply(velocity_cnn, (I_sample, t_sample), ps_, st) # shape: (28, 28, 1, N_b)
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

#             # Store the current parameters before update
#             old_ps = copy(ps)

#             # Compute gradients using the loss function closure
#             gs_tuple = gradient(loss_fn_closure, ps)
#             # Unpack the tuple to get the actual gradient
#             gs = gs_tuple[1]
#             gradient_clip_value = 1.0  # Example clip value
#             clipped_grads = clamp.(gs, -gradient_clip_value, gradient_clip_value)
#             # Check if gradients are very small or NaN
#             println("Gradient norms: ", norm(clipped_grads))

#             # Update the parameters using the optimizer
#             opt, ps = Optimisers.update!(opt, ps, clipped_grads)

#             # Calculate how much the parameters changed
#             ps_diff = norm(ps .- old_ps)
#             println("Parameter change: ", ps_diff)

#             # Calculate and display the mean loss for the batch
#             mean_loss = loss_fn_closure(ps)
#             println("Batch loss: $mean_loss")
#         end
#     end
#     return ps, st
# end