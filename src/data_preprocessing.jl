# Import libraries
using CSV, DataFrames
using LinearAlgebra
using Base
using Statistics


function load_data(filename)
  """
  Load stock price data from a CSV file.

  Args:
      filename: Name of the CSV file containing the stock price data.

  Returns:
      df: DataFrame containing the loaded stock price data.
  """
  # Get the current file path
  current_file = @__FILE__

  # Get the parent folder path
  parent_folder = dirname(dirname(current_file))

  # Join the folder and filename to create a path
  filepath = joinpath(parent_folder, "data", filename)

  df = CSV.read(filepath, DataFrame)
  return df
end


function generate_sequences(data, seq_length)
    """
    Generate input and target sequences from the given data.

    Args:
        data: Time series data.
        seq_length: Length of the input sequences.

    Returns:
        input_seq_array: List of input sequences.
        target_array: List of corresponding target sequences.
    """
    input_seqs = []
    target_seqs = []

    for i in 1:size(data, 1) - seq_length
        input_seq = data[i:i + seq_length - 1, :]
        target_seq = data[i + seq_length, :]
        push!(input_seqs, input_seq)
        push!(target_seqs, target_seq)
    end

    input_seq_array = permutedims(cat(input_seqs...; dims=3), (2,3,1))
    target_array = zeros(1, length(target_seqs), seq_length)
    target_array[1, :, end] = reshape(hcat(target_seqs...), length(target_seqs))

    return input_seq_array, target_array
end


function process_data(filename, split_ratio, seq_length)
    """
    Process the stock price data from a CSV file.

    Args:
        filename: Name of the CSV file containing the stock price data.
        split_ratio: Ratio of data to use for training (between 0 and 1).
        seq_length: Length of the input sequences.

    Returns:
        x_train: Input sequences for training.
        y_train: Target sequences for training.
        x_test: Input sequences for testing.
        y_test: Target sequences for testing.
        mean_value: Mean value for normalization.
        std_value: Std deviation for normalization.
    """
    # Load the CSV file
    df = load_data(filename)

    # Extract the closing prices
    prices = df[:, :Close]

    # Split the data into training and testing sets
    n = size(prices, 1)
    split_idx = Int(round(n * split_ratio))
    train_data = prices[1:split_idx]
    test_data = prices[split_idx+1:end]

    # Standardize the prices
    mean_value = mean(train_data)
    std_value = std(train_data)
    train_data = (train_data .- mean_value) ./ std_value
    test_data = (test_data .- mean_value) ./ std_value

    # Generate input and target sequences
    x_train, y_train = generate_sequences(train_data, seq_length) # (input_size, n_samples, total_grus)
    x_test, y_test = generate_sequences(test_data, seq_length)

    return x_train, y_train, x_test, y_test, mean_value, std_value
end