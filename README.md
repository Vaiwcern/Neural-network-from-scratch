# Neural-Network-From-Scratch

This project involves building a neural network from scratch using C++ and CUDA for parallel computation. The aim is to provide an in-depth understanding of neural network principles and how they can be implemented efficiently in C++.

## Structure

The repository is organized into several directories, each representing a different version or implementation of the neural network. Make sure you navigate to the appropriate directory before building the version you want to run.

## Prerequisites

- Ensure you have a compatible C++ compiler installed.
- NVIDIA's CUDA Toolkit must be installed for GPU acceleration.

## Preparation

Before running any version of the application, download the necessary data:

1. Open your terminal.
2. Run the data download script to ensure all necessary datasets are prepared:
   ```bash
   sh download_data.sh
   ```

## How to Run

To run different versions of the neural network, navigate into the respective directories using the `cd` command and compile the code using `make`.

### Steps

1. Navigate to the directory of the version you wish to run:
   ```bash
   cd gpu_ver1
   ```
2. Compile the code using the `make` command, which also executes the program:
   ```bash
   make
   ```