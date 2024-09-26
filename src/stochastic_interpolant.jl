# Define the stochastic interpolant function
function stochastic_interpolant(image1, image2, λ) 
    return cos.(π/2 .*λ) .* image1 .+ sin.(π/2 .*λ).* image2 
    # (1 .- λ) .* image1 .+ λ .* image2
end

### Plot an example of the stochastic interpolant ###
function visualize_interpolation(gaussian_images, target_images, image_index=1)
    # Extract the 2D images (Gaussian and Target)
    img1 = gaussian_images[image_index, :, :]  # Extract Gaussian image
    img2 = target_images[image_index, :, :]    # Extract Target image
    
    # Perform stochastic interpolation
    interpolated_img = stochastic_interpolant(img1, img2, 0.9) #rand()
    
    # Create plots for the Gaussian image, target image, and interpolated image
    p1 = heatmap(img1, color=:grays, axis=false, legend=false, title="Gaussian Image")
    p2 = heatmap(img2, color=:grays, axis=false, legend=false, title="Target Image")
    p3 = heatmap(interpolated_img, color=:grays, axis=false, legend=false, title="Interpolated Image")
    
    # Display the plots side by side
    plot(p1, p2, p3, layout=(1, 3), size=(900, 300))
end

### Time derivative stochastic interpolant ###
function time_derivative_stochastic_interpolant(image1, image2, λ)
    return -π/2 .*sin.(π/2 .*λ) .* image1 .+ π/2 .*cos.(π/2 .*λ).* image2 
    # -image1 .+ image2
end