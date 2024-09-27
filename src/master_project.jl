module master_project

# Write your package code here.
include("data.jl")
include("stochastic_interpolant.jl")
# include("cnn.jl")
include("train.jl")
include("generate_digit.jl")
include("u-net.jl")

# Add functions for data 
export reshape_mnist_data
export load_mnist_data
export plot_images
export load_mnist_labels
export generate_gaussian_images
export select_mnist_images
export resize_images

# Add functions for stochastic interpolant
export stochastic_interpolant
export visualize_interpolation
export time_derivative_stochastic_interpolant

# Add functions for u-net. 
export sinusoidal_embedding
export ResidualBlock
export DownBlock
export UpBlock
export UNet
export build_full_unet

# # Add the CNN
# export build_NN

# Add fucntions for training
export loss_fn
export get_minibatch
export train!

# Add functions for generating a digit
export forward_euler
export runge_kutta_4
export generate_digit
export plot_generated_digits

end
