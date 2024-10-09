# Define the stochastic interpolant function
function stochastic_interpolant(image1, image2, z, λ) 
    return cos.(π/2 .*λ) .* image1 .+ sin.(π/2 .*λ).* image2 # .+ .√(2 .* λ .* (1 .- λ)) .* z
end

### Plot an example of the stochastic interpolant ###
function visualize_interpolation(gaussian_images, target_images, image_index=1)
    # Extract the 2D images (Gaussian and Target)
    img1 = gaussian_images[image_index, :, :]  # Extract Gaussian image
    img2 = target_images[image_index, :, :]    # Extract Target image
    # Extract noise for the stochastic_interpolant
    z = randn(Float32, 32, 32)
    
    # Perform stochastic interpolation
    interpolated_img_noise = stochastic_interpolant(img1, img2, z, 0.7)
    interpolated_img = stochastic_interpolant(img1, img2, 0, 0.7)

    # Create plots for the Gaussian image, target image, and interpolated image
    p1 = heatmap(img1, color=:grays, axis=false, legend=false, title="Gaussian Image")
    p2 = heatmap(img2, color=:grays, axis=false, legend=false, title="Target Image")
    p3 = heatmap(interpolated_img, color=:grays, axis=false, legend=false, title="Interpolated Image")
    p4 = heatmap(interpolated_img_noise, color=:grays, axis=false, legend=false, title="Interpolated with noise")
    
    # Display the plots side by side
    plot(p1, p2, p3, p4, layout=(2, 2), size=(900, 900))
end

### Time derivative stochastic interpolant ###
function time_derivative_stochastic_interpolant(image1, image2, z, λ)
    return -π/2 .*sin.(π/2 .*λ) .* image1 .+ π/2 .*cos.(π/2 .*λ).* image2 # .+ ((2 .- 4 .* λ) ./ (2 .* .√(2 .* λ .* (1 .- λ)))) .* z
end