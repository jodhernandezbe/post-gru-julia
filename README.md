# Gated Recurrent Neural Network from Scratch in Julia

This is a personal project I created for two reasons:

<ol>
  <li>To write a Medium post</li>
  <li>To Learn and explore Julia programming language</li>
</ol>

In this project, I build a Recurrent Neural Network (RNN) from scratch using Julia. The RNN has Gated Recurrent Units (GRUs). The idea is to build the RNN to predict stock prices over time.

## Project structure

```bash
.
├── data
│   ├── AAPL.csv
│   ├── GOOG.csv
│   └── IBM.csv
├── plots
│   ├── residual_plot.png
│   └── sequence_plot.png
├── Project.toml
├── .pre-commit-config.yaml
├── src
│   ├── data_preprocessing.jl
│   ├── main.jl
│   ├── prediction_plots.jl
│   └── scratch_gru.jl
└── tests (unit testing)
    ├── test_data_preprocessing.jl
    ├── test_main.jl
    └── test_scratch_gru.jl
```

## Prerequisites

You should install Julia (version 1.9.1) in your laptop. Check the following link out:

[Platform Specific Instructions for Official Binaries](https://julialang.org/downloads/platform/)

### Packages:
- CSV (0.10.11)
- DataFrames (1.5.0)
- LinearAlgebra (standard)
- Base (standard)
- Statistics (standard)
- ArgParse (1.1.4)
- Plots (1.38.16)
- Random (standard)
- Test (standard, unit testing only)

As you can see in the project tree, there is a file named ```Project.toml```. You can use this file for installing the requiered dependencies by running the following command:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
This creates an environment with the requiered dependencies.

## Use

You are free to check each file inside the project directory. Nonetheless, I will give you here, some instructions in order to run the ```main.jl``` file.

### Help menu

To check the arguments received by this file via command-line, run the following command:

```bash
julia src/main.jl --help
```
It will print the following in your command line:

```bash
usage: main.jl [--data_file DATA_FILE] [--split_ratio SPLIT_RATIO]
               [--seq_length SEQ_LENGTH] [--hidden_size HIDDEN_SIZE]
               [--num_epochs NUM_EPOCHS]
               [--learning_rate LEARNING_RATE] [-h]

optional arguments:
  --data_file DATA_FILE
                        File containing the stock prices (default:
                        "GOOG.csv")
  --split_ratio SPLIT_RATIO
                        Ratio to split the dataset into training and
                        testing (type: Float64, default: 0.7)
  --seq_length SEQ_LENGTH
                        Lenght to form the sequences (also # of GRUs)
                        (type: Int64, default: 10)
  --hidden_size HIDDEN_SIZE
                        Hidden size for the GRUs (type: Int64,
                        default: 20)
  --num_epochs NUM_EPOCHS
                        Number of epochs for training (type: Int64,
                        default: 30)
  --learning_rate LEARNING_RATE
                        Learning rate for the training (type: Float64,
                        default: 0.0001)
  -h, --help            show this help message and exit

```

Look at how you can explore each of the arguments, considering their data type. In case you would like to make an analysis using a different stock, you have to include it in the ```data``` folder in .csv format.

### Run the code

```bash
julia src/main.jl --data_file GOOG.csv --split_ratio 0.7 --seq_length 10 --hidden_size 70 --num_epochs 1000 --learning_rate 0.00001 --project
```
