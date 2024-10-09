module master_project

# Write your package code here.
include("data.jl")
include("stochastic_interpolant.jl")
# include("cnn.jl")
# include("train_ODE.jl")
include("train_SDE.jl")
# include("generate_digit_ODE.jl")
include("generate_digit_SDE.jl")
# include("u-net_residual.jl")
# include("u-net_conv.jl")
include("u-net_conv_conditioned.jl")

# Add functions for data 
export reshape_mnist_data
export load_mnist_data
export plot_images
export load_mnist_labels
export generate_gaussian_images
export reshape_images
export filter_mnist_data
export labels_to_images

# Add functions for stochastic interpolant
export stochastic_interpolant
export visualize_interpolation
export time_derivative_stochastic_interpolant

# # Add the CNN
# export build_NN

# # Add functions for first u-net - residual
# export sinusoidal_embedding
# export ResidualBlock
# export DownBlock
# export UpBlock
# export UNet
# export build_full_unet
# export ConvNextBlock

# # Add functions for second u-net - convolutional
# export sinusoidal_embedding
# export ConvNextBlock
# export DownBlock
# export UpBlock
# export UNet
# export build_full_unet

# Add functions for third u-net - conditioning
export sinusoidal_embedding
export ConvNextBlock
export DownBlock
export UpBlock
export UNet
export build_full_unet

# # Add fucntions for training - ODE
# export loss_fn
# export get_minibatch
# export train!

# Add functions for training - SDE
export loss_fn
export get_minibatch
export train!

# # Add functions for generating a digit - ODE
# export forward_euler
# export runge_kutta_4
# export generate_digit
# export plot_generated_digits

# Add functions for generating a digit - SDE
export gamma
export alpha
export beta
export derivative_alpha
export derivative_beta
export epsilon
export euler_maruyama
export generate_digit
export compute_score_denoiser
export compute_score_velocity
export plot_generated_digits

end
