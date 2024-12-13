# Evaluation metrices: MSE
function mean_squared_error(y, y_pred; dev)
    y = y |> dev
    y_pred = y_pred |> dev
    mse = mean((y .- y_pred).^2)
    return mse
end

# Relative MSE
function mean_relative_mse(y, y_pred; epsilon=1e-8, dev)
    y = y |> dev
    y_pred = y_pred |> dev
    mse = mean_squared_error(y, y_pred; dev)
    variance_y = mean((y .- mean(y)).^2) + epsilon
    rel_mse = mse / variance_y
    return rel_mse
end

# RSME
function relative_rmse(y, y_pred; epsilon=1e-8, dev)
    y = y |> dev
    y_pred = y_pred |> dev
    rel_mse = mean_relative_mse(y, y_pred; epsilon, dev)
    rel_rmse = sqrt(rel_mse)
    return rel_rmse
end

function compute_metrics_average(y_true, y_pred; epsilon=1e-8, dev)
    y_true = y_true |> dev
    y_pred = y_pred |> dev

    num_samples = size(y_true, 4)

    mse = Float32[]         
    rel_mse = Float32[]     
    rel_rmse = Float32[]    

    for i in 1:num_samples
        y = y_true[:, :, :, i]
        y_hat = y_pred[:, :, :, i]
        
        # Compute sample-wise metrics
        push!(mse, mean_squared_error(y, y_hat; dev))
        push!(rel_mse, mean_relative_mse(y, y_pred; epsilon, dev))
        push!(rel_rmse, relative_rmse(y, y_pred; epsilon, dev))
    end

    # Compute averages
    avg_mse = mean(mse)
    avg_rel_mse = mean(rel_mse)
    avg_rel_rmse = mean(rel_rmse)

    println("Average Mean Squared Error (MSE): $avg_mse")
    println("Average Relative MSE: $avg_rel_mse")
    println("Average Relative RMSE: $avg_rel_rmse")

    return avg_mse, avg_rel_mse, avg_rel_rmse
end