using master_project

using Lux
using Random
using ComponentArrays
using Optimisers
# using LuxCUDA
using Statistics


# Choose CPU or GPU
dev = cpu_device() # gpu_device()

# Load MNIST data
train_images = load_mnist_data("mnist_data/train-images.idx3-ubyte", 60000, 28, 28);
train_labels = load_mnist_labels("mnist_data/train-labels.idx1-ubyte", 60000);
plot_images(train_images, 9);

# Filter images for the digit or take everything
digit = nothing;
num_samples = 2000;
train_images_filtered, train_labels_filtered = filter_mnist_data(train_images, train_labels, digit, num_samples);

# reshape images to size 32 by 32
train_images_reshaped = reshape_images(train_images_filtered, 32);
# standardize the filtered mnist images --> mean 0 and stv 1.
train_images_filtered = (train_images_reshaped .- mean(train_images_reshaped)) ./ std(train_images_reshaped);

plot_images(train_images_filtered, 9);

# Convert the integer labels to images of 32 by 32. 
label_images_32x32 = labels_to_images(Int.(train_labels_filtered), 32);

# Plot x gaussian images to check if the process went correctly 
train_gaussian_images = generate_gaussian_images(num_samples, 32, 32);

# Stochastic interpolant using time 't=0.9', with and without noise term. 
visualize_interpolation(train_gaussian_images, train_images_filtered)

# Define the network - u-net with ConvNextBlocks 
velocity_cnn = build_full_unet(16,[32,64,128],8);

# Initialize the network parameters and the state
ps, st = Lux.setup(Random.default_rng(), velocity_cnn); # .|> dev;

# Define the batch_size
batch_size = 32;
num_batches = ceil(Int, num_samples / batch_size);
num_epochs = 10;

# Define the Adam optimizer with a learning rate (1e-6)
opt = Optimisers.setup(Adam(1.0e-3, (0.9f0, 0.99f0), 1e-10), ps);

# train_images_filtered = train_images_filtered |> dev;
# train_gaussian_images = train_gaussian_images |> dev;

# Start training
train!(velocity_cnn, ps, st, opt, num_epochs, batch_size, train_gaussian_images, train_images_filtered, label_images_32x32, num_batches, dev);

# Generate digits 
gaussian_image = randn(Float32, 32, 32, 1, batch_size);  # Generate a initial Gaussian noise image
num_steps = 200;  # Number of steps to evolve the image
step_size = 1.0 / num_steps;  # Step size (proportional to time step)
label = 9; # Choose the digit you want to generate. 

# Generate the digit image
dev = cpu_device()
gaussian_image = gaussian_image |> dev;
_st = Lux.testmode(st);
generated_digit = generate_digit(velocity_cnn, ps, _st, gaussian_image, label, num_steps, batch_size, dev; method=:euler);

generated_digit = generated_digit |> cpu_device();
gaussian_image = gaussian_image |> cpu_device();

# Show the first 9 generated images
plot_generated_digits(generated_digit, 9);

