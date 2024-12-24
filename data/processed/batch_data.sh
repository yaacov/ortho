#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <directory_name> <batch_size>"
    exit 1
fi

# Get the arguments
input_dir="$1"
batch_size="$2"

# Validate the directory
if [ ! -d "$input_dir" ]; then
    echo "Error: Directory $input_dir does not exist."
    exit 1
fi

# Validate the batch size
if ! [[ "$batch_size" =~ ^[0-9]+$ ]] || [ "$batch_size" -le 0 ]; then
    echo "Error: Batch size must be a positive integer."
    exit 1
fi

# Create output directories and copy files in batches
counter=1
file_count=0
mkdir -p "${input_dir}_${counter}"

for file in "$input_dir"/*; do
    # Skip if no files match the glob
    if [ ! -e "$file" ]; then
        echo "No files found in $input_dir."
        exit 0
    fi

    # Copy the file to the current batch directory
    cp "$file" "${input_dir}_${counter}/"
    file_count=$((file_count + 1))

    # Check if the batch is full
    if [ "$file_count" -eq "$batch_size" ]; then
        counter=$((counter + 1))
        mkdir -p "${input_dir}_${counter}"
        file_count=0
    fi
done

echo "Files have been copied into batch directories."
