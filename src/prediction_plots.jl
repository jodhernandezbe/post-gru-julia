# Import libraries
using Plots

function plot_seq_predictions(y_train, y_test, y_train_hat, y_test_hat)
    """
    Plot the sequential values including y_train, y_test, and y_test_hat.

    Args:
        y_train: Target values for the training set.
        y_test: Target values for the testing set.
        y_train_hat: List of predicted values for the training test.
        y_test_hat: List of predicted values for the testing set.

    Returns:
        Nothing. The figure is saved at the specified savepath.
    """
    # Plot real values
    plot(range(1, size(y_train,2), length=size(y_train,2)),
            y_train[1, :, end],
            label="y_train",
            seriestype=:scatter,
            ms=3, ma=0.5)
    plot!(range(size(y_train,2), size(y_train,2) + size(y_test,2), length=size(y_test,2)),
            y_test[1, :, end],
            label="y_test",
            seriestype=:scatter,
            ms=3, ma=0.5)
    
    # Plot predicted values
    plot!(range(1, size(y_train_hat,2), length=size(y_train_hat,2)),
            y_train_hat[1, :, end],
            label="y_train_hat",
            linewidth=2)
    plot!(range(size(y_train_hat,2), size(y_train_hat,2) + size(y_test_hat,2), length=size(y_test_hat,2)),
            y_test_hat[1, :, end],
            label="y_test_hat",
            linewidth=2)

    # Set the legend
    plot!(legend=:outertop, legendcolumns=4)

    # Resize the plot
    plot!(size = (800, 600))

    # Add axis labels
    xlabel!("Sample", fontweight = :bold, fontfamily = "sans-serif")
    ylabel!("Price [USD]", fontweight = :bold, fontfamily = "sans-serif")
    
    # Save the figure
    current_file = @__FILE__
    parent_folder = dirname(dirname(current_file))
    filepath = joinpath(parent_folder, "plots", "sequence_plot.png")
    savefig(filepath)
end


function plot_residuals(y_train, y_train_hat, y_test, y_test_hat)
    """
    Create a residual plot to visualize the differences between the predicted values and the actual values.

    Args:
        y_train: Actual target values for the training set.
        y_train_hat: Predicted values for the training set.
        y_test: Actual target values for the testing set.
        y_test_hat: Predicted values for the testing set.

    Returns:
        Nothing. The figure is saved as 'residual_plot.png'.
    """
    # Calculate residuals
    residuals_train = y_train[1, :, end] - y_train_hat[1, :, end]
    residuals_test = y_test[1, :, end] - y_test_hat[1, :, end]
    
    scatter(y_train_hat[1, :, end],
            residuals_train,
            label="Train set",
            ms=3, ma=0.5)
    scatter!(y_test_hat[1, :, end],
            residuals_test,
            label="Test set",
            ms=3, ma=0.5)

    # Add zero residual line
    hline!([0], line=:dash, color=:purple, label="Zero residual")

    # Resize the plot
    plot!(size = (800, 600))

    # Set the legend
    plot!(legend=:outertop, legendcolumns=3)

    # Set plot labels
    xlabel!("Predicted value [USD]")
    ylabel!("Residual [USD]")

    # Save the figure
    current_file = @__FILE__
    parent_folder = dirname(dirname(current_file))
    filepath = joinpath(parent_folder, "plots", "residual_plot.png")
    savefig(filepath)
end


function plot_costs(cost_per_epoch)
    """
    Plots a graph of cost per epoch during model fitting process

    Args:
        cost_per_epoch : A vector containing costs at each iteration
    
    Returns:
        Nothing. The figure is saved as 'cost_plot.png'.
    """
    total_costs = size(cost_per_epoch, 1)
    plot(range(1, total_costs, length=total_costs),
            cost_per_epoch,
            linewidth=2,
            label="Training cost")


    # Resize the plot
    plot!(size = (800, 600))

    # Add axis labels
    xlabel!("Epoch", fontweight = :bold, fontfamily = "sans-serif")
    ylabel!("Cost", fontweight = :bold, fontfamily = "sans-serif")
    
    # Save the figure
    current_file = @__FILE__
    parent_folder = dirname(dirname(current_file))
    filepath = joinpath(parent_folder, "plots", "cost_plot.png")
    savefig(filepath)
end