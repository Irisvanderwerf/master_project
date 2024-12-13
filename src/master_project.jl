module master_project

# include("data_mnist.jl")
# include("data_navierstokes_spectral.jl")
include("data_navierstokes_FVM.jl")
include("stochastic_interpolant.jl")
include("train.jl")
include("generate_closure.jl")
include("evaluation.jl")
# include("generate_digit_SDE.jl")
# include("cnn.jl")
# include("u-net_residual.jl")
# include("u-net_conv.jl")
# include("u-net_conv_conditioned.jl")
include("u-net_conv_cond_attention.jl")
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
export face_average_syver
export face_average_syver!
export compute_mean_std
export standardize_training_set_per_channel
export inverse_standardize_set_per_channel
export generate_or_load_data
export generate_or_load_standardized_data
export print_min_max
export compute_velocity_magnitude
export plot_velocity_magnitudes
export create_training_sets

# # Add functions for data 
# export reshape_mnist_data
# export load_mnist_data
# export plot_images
# export load_mnist_labels
# export generate_gaussian_images
# export reshape_images
# export filter_mnist_data
# export labels_to_images

# Add functions for stochastic interpolant
export stochastic_interpolant
export visualize_interpolation
export time_derivative_stochastic_interpolant

# # Add the CNN
# export build_NN

export rk4_with_closure_deterministic
export inference_deterministic

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
# export ConvNextBlock_up
# export ConvNextBlock_down
export UNet
export build_full_unet

export gpu_attention_broadcast
export SelfAttentionBlock
export ConvNextBlock_down_with_attention
export ConvNextBlock_up_with_attention

# Add functions for training - ODE or SDE
export loss_fn
# export get_minibatch_MNIST
export get_minibatch_NS
export train!
export save_model
export load_model
export initialize_or_load_model
export relative_root_mse
export train_deterministic!

# Add functions for generating closure - ODE or SDE
export forward_euler
export runge_kutta_4
# export generate_digit
# export plot_generated_digits
export generate_closure
export gamma
export alpha
export beta
export derivative_alpha
export derivative_beta
export epsilon
export euler_maruyama
export compute_score_denoiser
export compute_score_velocity
export inference
# export compute_energy_spectrum
# export radial_binning
export compute_energy_spectra
export compute_total_energy
# export runge_kutta_4_closure_term
export rk4_with_closure
export inference_deterministic_without_ground_truth

# evaluation of the network
export mean_squared_error
export mean_relative_mse
export relative_rmse
export compute_metrics_average

end

