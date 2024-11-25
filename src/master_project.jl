module master_project

# Write your package code here.
# include("data_navierstokes_spectral.jl")
include("data_navierstokes_FVM.jl")
# include("data_mnist.jl")
include("stochastic_interpolant.jl")
# include("cnn.jl")
include("train_ODE.jl")
# include("train_SDE.jl")
include("generate_digit_ODE.jl")
# include("generate_digit_SDE.jl")
# include("u-net_residual.jl")
# include("u-net_conv.jl")
include("u-net_conv_conditioned.jl")
# include("u-net_conv_conditioned_complex.jl")

# # Add functions for Navier Stokes simulations - spectral decomposition
# export Q
# export F
# export project
# export step_rk4
# export vorticity
# export gaussian
# export create_spectrum
# export random_field
# export create_params
# export zeros
# export standardize_training_set_per_channel

# Add functions for Navier Stokes simulations - Finite Volume Method (library from Syver)
export step_rk4
export face_averaging_velocity_2D
export compute_mean_std
export standardize_training_set_per_channel
export inverse_standardize_set_per_channel

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

# Add functions for fourth u-net - conditioning & handle complex and real numbers seperately 
export sinusoidal_embedding
export ConvNextBlock_up
export ConvNextBlock_down
export UNet
export build_full_unet

# Add functions for training - ODE or SDE
export loss_fn
# export get_minibatch_MNIST
export get_minibatch_NS
export train!
export save_model
export load_model
export initialize_or_load_model

# Add functions for generating a digit - ODE
export forward_euler
export runge_kutta_4
export generate_digit
export plot_generated_digits
export generate_closure

# # Add functions for generating a digit - SDE
# export gamma
# export alpha
# export beta
# export derivative_alpha
# export derivative_beta
# export epsilon
# export euler_maruyama
# export generate_digit
# export generate_closure
# export compute_score_denoiser
# export compute_score_velocity

end
