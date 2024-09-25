module master_project

# Write your package code here.
include("data.jl")
include("stochastic_interpolant.jl")
include("cnn.jl")
include("train.jl")
include("generate_digit.jl")

# Add functions for data 
export reshape_mnist_data
export load_mnist_data
export plot_images
export load_mnist_labels
export generate_gaussian_images
export filter_train_images_by_digit

# Add functions for stochastic interpolant
export stochastic_interpolant
export visualize_interpolation
export time_derivative_stochastic_interpolant

# Add the CNN
export build_NN
export UNet

# Add fucntions for training
export loss_fn
export get_minibatch
export train!

# Add functions for generating a digit
export generate_digit
export plot_generated_digits

end
