using Plots

function generate_digit(velocity_cnn, ps, st, initial_gaussian_image, num_steps, batch_size, step_size, dev)
    images = initial_gaussian_image
    
    t_range = LinRange(0, 1, num_steps)
    dt = Float32(t_range[2] - t_range[1])

    # Simulate forward evolution over time from t = 0 to t = 1
    for i in 1:num_steps-1
        # Compute the current time t in the interval [0, 1]

        t = t_range[i]
        
        # Reshape t_sample to match the batch size
        t_sample = Float32.(fill(t,(1,1,1,batch_size))) |> dev
        # t_sample = Float32.(reshape([t], 1, 1, 1, batch_size))  # For single image generation
        
        # Predict the velocity field using the neural network
        velocity, st = Lux.apply(velocity_cnn, (images, t_sample), ps, st)

        
        # Update the image based on the velocity field
        images = images .+ dt .* velocity
    end
    # Maybe add clamping 
    return clamp.(images,0.0,1.0)
end

### Plot multiple generated images ###
function plot_generated_digits(images, num_images_to_show)
    num_cols = ceil(Int, sqrt(num_images_to_show))
    num_rows = ceil(Int, num_images_to_show / num_cols)

    p = plot(layout=(num_rows, num_cols), size=(800, 800))
    
    for i in 1:num_images_to_show
        img = reshape(images[:, :, 1, i], (28, 28))  # Reshape to (28, 28)
        heatmap!(img, color=:grays, axis=false, legend=false, subplot=i, title="Generated Image $i")
    end
    
    display(p)
end