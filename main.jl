using master_project

using Lux
using Random
using ComponentArrays
using Optimisers 

# Load MNIST data
train_images = load_mnist_data("mnist_data/train-images.idx3-ubyte", 60000, 28, 28)
train_labels = load_mnist_labels("mnist_data/train-labels.idx1-ubyte", 60000)
plot_images(train_images, 9)

# Filter images for the digit '5'
digit = 3
num_samples = 1024
train_images_filtered = filter_train_images_by_digit(train_images, train_labels, digit, num_samples)
plot_images(train_images_filtered, 9)

# Plot x gaussian images to check if the process went correctly 
train_gaussian_images = generate_gaussian_images(num_samples, 28, 28)

# Stochastic interpolant
visualize_interpolation(train_gaussian_images, train_images_filtered)

# Define the network
velocity_cnn = build_NN()

# Initialize the network parameters and the state
ps, st = Lux.setup(Random.default_rng(), velocity_cnn)
ps = ComponentArray(ps)

# Define the batch_size
batch_size = 32
num_batches = ceil(Int, num_samples / batch_size)
num_epochs = 10

# Define the Adam optimizer with a learning rate (1e-6)
opt = Optimisers.setup(Adam(1.0e-4),ps)

# Start training
train!(velocity_cnn, ps, st, opt, num_epochs, batch_size, train_gaussian_images, train_images_filtered, num_batches)

# Generate digits 
gaussian_image = randn(Float32, 28, 28, 1, batch_size)  # Generate a initial Gaussian noise image
num_steps = 100  # Number of steps to evolve the image
step_size = 1.0 / num_steps  # Step size (proportional to time step)
# Generate the digit image

_st = Lux.testmode(st)
generated_digit = generate_digit(velocity_cnn, ps, _st, gaussian_image, num_steps, batch_size, step_size)
# Show the first 9 generated images
plot_generated_digits(generated_digit, 9)
