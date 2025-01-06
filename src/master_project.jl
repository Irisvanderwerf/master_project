module master_project

include("data_navierstokes_FVM.jl")
include("stochastic_interpolant.jl")
include("u-net_conv_cond_attention.jl")
include("train.jl")
include("generate_closure.jl")
include("evaluation.jl")
include("inference.jl")

export step_rk4
export face_average_syver
export face_average_syver!
export compute_mean_std
export standardize_training_set_per_channel
export standardize_trajectories

export inverse_standardize_set_per_channel
export save_large_bson
export load_large_bson
export generate_or_load_data
export generate_or_load_standardized_data
export compute_velocity_magnitude
export plot_velocity_magnitudes
export create_training_sets

export stochastic_interpolant
export time_derivative_stochastic_interpolant
export alpha
export beta
export sigma
export derivative_alpha
export derivative_beta
export derivative_sigma

export periodic_padding_gpu
export ConvPeriodicLayer
export SelfAttentionBlock
export sinusoidal_embedding
export ConvNextBlock_down
export ConvNextBlock_up
export BottomLayerWithAttention
export UNet
export build_full_unet

export initialize_or_load_model
export load_model
export train!
export get_minibatch_NS
export loss_fn
export save_model

export generate_closure
export generate_closure_with_tunable_diffusion
export diffusion
export score
export euler_maruyama
export generate_closure_test_trajectory
export generate_closure_test_trajectory_tunable_drift

export compute_metrics_average
export mean_squared_error
export mean_relative_mse
export relative_rmse

export inference
export compute_energy_spectra
export inference_tunable_diffusion
export inference_mean

end

