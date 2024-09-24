using Plots

### MNIST data  - images ###
function reshape_mnist_data(data::Vector{UInt8}, num_images::Int, num_rows::Int, num_cols::Int)
    # Skip the first 16 bytes (header)
    data = data[17:end]
    # Reshape the remaining data vector into a 3D array (num_images, num_rows, num_cols)
    images = reshape(data, (num_cols, num_rows, num_images))
    # Permute dimensions to (num_images, num_rows, num_cols)
    images = permutedims(images, (3, 2, 1))
    # Normalize the images by dividing by 255.0
    images = Float32.(images) ./ 255.
    return images
end

function load_mnist_data(train_path::String, num_images_train::Int, num_rows::Int, num_cols::Int)
    # Load raw data
    x_train = read(train_path)
    # Reshape and normalize the training and test images
    train_images = reshape_mnist_data(x_train, num_images_train, num_rows, num_cols)
    return train_images
end

### MNIST data  - labels ###
function load_mnist_labels(label_path::String, num_labels::Int)
    labels = read(label_path)[9:end]  # Skip the first 8 bytes (header)
    return labels[1:num_labels]
end

### MNIST data - digit specific data ###
function filter_train_images_by_digit(images, labels, target_digit, max_count=nothing)
    # Get indices of images corresponding to the target digit
    indices = findall(labels .== target_digit)
    
    # If max_count is specified, limit the number of images returned
    if !isnothing(max_count) && length(indices) > max_count
        indices = indices[1:max_count]
    end
    
    # Select the corresponding images
    filtered_images = images[indices, :, :, :]
    
    return filtered_images[:,:,:,1]
end

### Initial distribution data - Gaussian (Normal distributed) images ###
function generate_gaussian_images(num_images::Int, num_rows::Int, num_cols::Int, mean::Float64=0.0, stddev::Float64=1.0)
    # Create a 3D array to hold the images (num_images, num_rows, num_cols)
    images = randn(Float32, num_images, num_rows, num_cols) .* stddev .+ mean
    return images
end

### Plot the images ###
function plot_images(images, num_images_to_show)
    # Calculate the layout dimensions (rows and columns)
    num_cols = ceil(Int, sqrt(num_images_to_show))
    num_rows = ceil(Int, num_images_to_show / num_cols)
    # Create a list of heatmaps to display the images
    p = plot(layout=(num_rows, num_cols), size=(800, 800))
    for i in 1:num_images_to_show
        img = Gray.(images[i, :, :])  # Convert to grayscale
        heatmap!(img, color=:grays, axis=false, legend=false, subplot=i)
    end
    display(p)
end