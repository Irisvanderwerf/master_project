function stochastic_interpolant(image1, image2, W, λ, ϵ) 
    return alpha(λ) .* image1 .+ beta(λ) .* image2 .+ sigma(λ, ϵ) .* W
end

function time_derivative_stochastic_interpolant(image1, image2, W, λ, ϵ)
    return  derivative_alpha() .* image1 .+ derivative_beta(λ) .* image2 .+ derivative_sigma(ϵ) .* W
end

function alpha(λ)
    alpha = 1 .- λ
    return alpha    
end

function beta(λ)
    beta = λ.^ 2
    return beta 
end

function sigma(λ, ϵ)
    sigma = ϵ .* (1 .- λ)
    return sigma
end

function derivative_alpha()
    deriv_alpha = -1
    return deriv_alpha  
end

function derivative_beta(λ)
    deriv_beta = 2 .* λ
    return deriv_beta   
end

function derivative_sigma(ϵ)
    deriv_sigma = -ϵ
    return deriv_sigma
end