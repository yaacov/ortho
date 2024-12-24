#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory_name>"
    exit 1
fi

# Get the target directory name
target_dir="$1"

# Check if the target directory already exists
if [ -d "$target_dir" ]; then
    echo "Error: Directory $target_dir already exists. Aborting to prevent overwriting."
    exit 1
fi

# Create the target directory
mkdir -p "$target_dir"
echo "Created directory: $target_dir"

# Initialize a flag to track whether any directories were processed
found_dirs=false

# Find all directories matching <dirname>_<number> pattern
for dir in "${target_dir}"_*; do
    # Skip if no directories match the pattern
    if [ ! -d "$dir" ]; then
        continue
    fi

    found_dirs=true

    # Copy all files from the subdirectory to the target directory
    for file in "$dir"/*; do
        if [ -f "$file" ]; then
            cp "$file" "$target_dir/"
        fi
    done

    # Delete the directory after copying its files
    rm -rf "$dir"
    echo "Deleted directory: $dir"
done

# Check if any directories were processed
if [ "$found_dirs" = false ]; then
    echo "No matching subdirectories found."
else
    echo "All files have been copied and all matching directories have been deleted."
fi
