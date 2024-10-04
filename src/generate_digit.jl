using Plots

# Forward Euler method
function forward_euler(velocity_cnn, ps, st, images, label, t, dt, batch_size, dev)
    # Reshape t_sample to match the batch size
    t_sample = Float32.(fill(t, (1, 1, 1, batch_size))) |> dev

    # Ensure images are Float32
    images = Float32.(images)

    # Predict the velocity field using the neural network
    velocity, st = Lux.apply(velocity_cnn, (images, t_sample, label), ps, st)

    # Update the images based on the velocity field
    updated_images = images .+ dt .* velocity

    return updated_images, st
end

function runge_kutta_4(velocity_cnn, ps, st, images, label,  t, dt, batch_size, dev)
    # Reshape t_sample to match the batch size
    t_sample = Float32.(fill(t, (1, 1, 1, batch_size))) |> dev

    # Ensure images are Float32
    images= Float32.(images)
    
    # Predict the velocity field using the neural network
    velocity, st = Lux.apply(velocity_cnn, (images, t_sample, label), ps, st)
    k1 = dt .* velocity

    t_sample_next = Float32.(fill(t + dt/2, (1, 1, 1, batch_size))) |> dev
    velocity_k2, st = Lux.apply(velocity_cnn, (images .+ k1 ./ 2, t_sample_next, label), ps, st)
    k2 = dt .* velocity_k2

    t_sample_next = Float32.(fill(t + dt/2, (1, 1, 1, batch_size))) |> dev
    velocity_k3, st = Lux.apply(velocity_cnn, (images .+ k2 ./ 2, t_sample_next, label), ps, st)
    k3 = dt .* velocity_k3

    t_sample_next = Float32.(fill(t + dt, (1, 1, 1, batch_size))) |> dev
    velocity_k4, st = Lux.apply(velocity_cnn, (images .+ k3, t_sample_next, label), ps, st)
    k4 = dt .* velocity_k4

    # Update the images based on the RK4 method
    updated_images = images .+ (k1 .+ 2k2 .+ 2k3 .+ k4) ./ 6
    
    return updated_images, st
end

# Main  function to generate digits
function generate_digit(velocity_cnn, ps, st, initial_gaussian_image, label, num_steps, batch_size, dev; method=:rk4)
    images = Float32.(initial_gaussian_image)  # Ensure initial images are Float32
    label = fill(label, 32, 32, 1, batch_size) # Set to proper size for the conditioning 
    
    t_range = LinRange(0, 1, num_steps)
    dt = t_range[2] - t_range[1]

    # Simulate forward evolution over time from t = 0 to t = 1
    for i in 1:num_steps-1
        # Compute the current time t in the interval [0, 1]
        t = t_range[i]
        
        # Choose the integration method
        if method == :euler
            images, st = forward_euler(velocity_cnn, ps, st, images, label, t, dt, batch_size, dev)
        elseif method == :rk4
            images, st = runge_kutta_4(velocity_cnn, ps, st, images, label, t, dt, batch_size, dev)
        else
            error("Unknown method: $method. Use :euler or :rk4.")
        end
    end

    # Maybe add clamping 
    return images # clamp.(images, 0.0, 1.0)
end

### Plot multiple generated images ###
function plot_generated_digits(images, num_images_to_show)
    num_cols = ceil(Int, sqrt(num_images_to_show))
    num_rows = ceil(Int, num_images_to_show / num_cols)

    p = plot(layout=(num_rows, num_cols), size=(800, 800))
    
    for i in 1:num_images_to_show
        img = reshape(images[:, :, 1, i], (32, 32))  # Reshape to (28, 28) - changed to (32,32)
        heatmap!(img, color=:grays, axis=false, legend=false, subplot=i, title="Generated Image $i")
    end
    
    display(p)
end
