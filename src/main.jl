# Import modules
include("data_preprocessing.jl")
include("prediction_plots.jl")
include("scratch_gru.jl")

# Import libraries
using ArgParse

function main(data_file, split_ratio, seq_length, hidden_size, num_epochs, learning_rate)
    """
    Main function to process data, train a GRU model, make predictions, evaluate the model, and create plots.

    Args:
        data_file: Path to the data file.
        split_ratio: Ratio to split the data into training and testing sets.
        seq_length: Length of input sequences.
        hidden_size: Size of the hidden layer in the GRU model.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for gradient descent.

    Returns:
        Nothing. Prints the mean squared error and creates prediction and residual plots.
    """
    # Process the data
    x_train, y_train, x_test, y_test, mean_value, std_value = process_data(data_file, split_ratio, seq_length)

    # Train the GRU model
    parameters, cost_per_epoch = train_gru(x_train, y_train, hidden_size, learning_rate, num_epochs)

    # Predict using the trained model
    y_train_hat = predict_gru(x_train, parameters)
    y_test_hat = predict_gru(x_test, parameters)

    # Evaluate the predictions (e.g., compute the mean squared error)
    output_size, n_samples, total_grus = size(y_test_hat)
    mse = sum((y_test_hat .- y_test).^2) / (output_size * n_samples * total_grus)
    println("Mean Squared Error: $mse")

    # Destandardize the data
    y_train = y_train .* std_value .+ mean_value
    y_test = y_test .* std_value .+ mean_value
    y_train_hat = y_train_hat .* std_value .+ mean_value
    y_test_hat = y_test_hat .* std_value .+ mean_value

    # Plot predictions
    plot_seq_predictions(y_train, y_test, y_train_hat, y_test_hat)

    # Plot residuals
    plot_residuals(y_train, y_train_hat, y_test, y_test_hat)

    # Plot cost
    plot_costs(cost_per_epoch)
end


function parse_commandline()
    """
    Function to create an command-line argument parsing
    """

    parser = ArgParseSettings()

    @add_arg_table parser begin
        "--data_file"
            help = "File containing the stock prices"
            default = "GOOG.csv"
            arg_type = String
        "--split_ratio"
            help = "Ratio to split the dataset into training and testing"
            default = 0.7
            arg_type = Float64
        "--seq_length"
            help = "Lenght to form the sequences (also # of GRUs)"
            default = 10
            arg_type = Int
        "--hidden_size"
            help = "Hidden size for the GRUs"
            default = 20
            arg_type = Int
        "--num_epochs" 
            help = "Number of epochs for training"
            default = 30
            arg_type = Int
        "--learning_rate"
            help = "Learning rate for the training"
            default = 0.0001
            arg_type = Float64
    end

    return parse_args(parser)
end


# Access the values of the parsed arguments
args = parse_commandline()
data_file = args["data_file"]
split_ratio = args["split_ratio"]
seq_length = args["seq_length"]
hidden_size = args["hidden_size"]
num_epochs = args["num_epochs"]
learning_rate = args["learning_rate"]

# Access to the main function
main(data_file, split_ratio, seq_length, hidden_size, num_epochs, learning_rate)