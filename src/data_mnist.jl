using Plots
using Images
using ImageTransformations

function reshape_mnist_data(data::Vector{UInt8}, num_images::Int, num_rows::Int, num_cols::Int)
    data = data[17:end]
    images = reshape(data, (num_cols, num_rows, num_images))
    images = permutedims(images, (3, 2, 1))
    images = Float32.(images) ./ 255.
    return images
end

function load_mnist_data(train_path::String, num_images_train::Int, num_rows::Int, num_cols::Int)
    x_train = read(train_path)
    train_images = reshape_mnist_data(x_train, num_images_train, num_rows, num_cols)
    return train_images
end

function load_mnist_labels(label_path::String, num_labels::Int)
    labels = read(label_path)[9:end]
    return labels[1:num_labels]
end

function filter_mnist_data(images, labels, digit=nothing, num_samples=nothing)
    if isnothing(digit)
        if isnothing(num_samples)
            return images, labels
        else
            indices = rand(1:size(images, 1), num_samples)
        end
    else
        indices = rand(findall(labels .== digit), num_samples)
    end
    return images[indices, :, :], labels[indices]
end

function reshape_images(images, target_size)
    num_images, height, width = size(images)
    padding = (target_size - height) ÷ 2
    padded_images = zeros(eltype(images), num_images, target_size, target_size)
    for i in 1:num_images
        padded_images[i, padding+1:padding+height, padding+1:padding+width] .= images[i, :, :]
    end
    
    return padded_images
end

function labels_to_images(train_labels::Vector{Int}, target_size::Int)
    num_samples = length(train_labels)
    label_images = zeros(Int, num_samples, target_size, target_size)
    for i in 1:num_samples
        label_images[i, :, :] .= train_labels[i] 
    end    
    return label_images
end

function generate_gaussian_images(num_images::Int, num_rows::Int, num_cols::Int, mean::Float64=0.0, stddev::Float64=1.0)
    images = randn(Float32, num_images, num_rows, num_cols) .* stddev .+ mean
    return images
end

function plot_images(images, num_images_to_show)
    num_cols = ceil(Int, sqrt(num_images_to_show))
    num_rows = ceil(Int, num_images_to_show / num_cols)
    p = plot(layout=(num_rows, num_cols), size=(800, 800))
    for i in 1:num_images_to_show
        img = Gray.(images[i, :, :])
        heatmap!(img, color=:grays, axis=false, legend=false, subplot=i)
    end
    display(p)
end



