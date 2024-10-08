using Plots
using Images
using ImageTransformations

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

## Filter the MNIST data ###
function filter_mnist_data(images, labels, digit=nothing, num_samples=nothing)
    if isnothing(digit)
        if isnothing(num_samples)
            # Return all images and labels if no digit or number of samples is specified
            return images, labels
        else
            # Randomly select num_samples from all images and labels
            indices = rand(1:size(images, 1), num_samples)
        end
    else
        # Select images and labels for the chosen digit
        indices = rand(findall(labels .== digit), num_samples)
    end
    # Return both the selected images and their corresponding labels
    return images[indices, :, :], labels[indices]
end

### Reshape the images to size 32 by 32 ###
function reshape_images(images, target_size)
    # Get the number of images and their original height and width
    num_images, height, width = size(images)
    
    # Calculate padding for top/bottom and left/right
    padding = (target_size - height) ÷ 2
    
    # Initialize an array of zeros for the padded images
    padded_images = zeros(eltype(images), num_images, target_size, target_size)
    
    # Loop through each image and place the 28x28 image in the center of the 32x32 padded image
    for i in 1:num_images
        padded_images[i, padding+1:padding+height, padding+1:padding+width] .= images[i, :, :]
    end
    
    return padded_images
end

### Function to transform labels into 32x32 images ###
function labels_to_images(train_labels::Vector{Int}, target_size::Int)
    num_samples = length(train_labels)
    # Initialize an array of size (num_samples, target_size, target_size)
    label_images = zeros(Int, num_samples, target_size, target_size)
    
    # Fill each 32x32 image with the corresponding label value
    for i in 1:num_samples
        label_images[i, :, :] .= train_labels[i]  # Fill the image with the label value
    end
    
    return label_images
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



