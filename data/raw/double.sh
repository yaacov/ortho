#!/bin/bash

# Ensure ImageMagick is installed
if ! command -v convert &> /dev/null; then
    echo "Error: ImageMagick is not installed. Please install it and try again."
    exit 1
fi

# List of image files to process
images=(
    "yemen0003.jpg"
    "001.png"
    "levontin_avoteinu_0003.jpg"
    "levontin_avoteinu_0015.jpg"
    "yemen0002.jpg"
    "docu0003.jpg"
    "1.jpg"
    "sheinkin_vol1_0015.jpg"
    "_014.png"
)

# Loop through each image file
for image in "${images[@]}"; do
    if [[ -f $image ]]; then
        # Extract the filename and extension
        filename="${image%.*}"
        extension="${image##*.}"

        # Generate the output filename
        output="${filename}_dbl.${extension}"

        # Resize the image to double its size and save it
        convert "$image" -resize 200% "$output"
        
        echo "Resized $image -> $output"
    else
        echo "Warning: File $image does not exist. Skipping."
    fi
done

echo "Processing complete."
