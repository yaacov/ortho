import os
import argparse

from utils.get_image_files import get_image_files
from utils.load_image import load_image
from utils.save_glyphs import save_glyphs


def process_image(image_path: str, output_dir: str) -> None:
    """
    Processes a single image by cleaning it and extracting connected glyphs.

    Args:
        image_path (str): Path to the input image.
        output_dir (str): Path to save the processed image.
    """
    # Load the cleaned image
    cleaned_image = load_image(image_path)
    base_name = os.path.basename(image_path)
    base_name_without_ext = os.path.splitext(os.path.basename(image_path))[0]

    glyphs_dir = os.path.join(output_dir, "glyphs")
    if not os.path.exists(glyphs_dir):
        os.makedirs(glyphs_dir)

    # Save connected components (e.g., characters) from the image
    save_glyphs(cleaned_image, glyphs_dir, f"{base_name_without_ext}_", 32)

    # Print a summary of the processed image
    print(f"Processed '{base_name}'")


def process_images_in_directory(
    input_dir: str, output_dir: str, max_images: int = None
) -> None:
    """
    Processes all images in the given directory.

    Args:
        input_dir (str): Directory containing input images.
        output_dir (str): Directory to save the processed output images.
        max_images (int, optional): Maximum number of images to process. If None, all images are processed.
    """
    # Get a list of images in the input directory
    images = get_image_files(input_dir, max_images)

    # Process each image in the directory
    for image_name in images:
        input_path = os.path.join(input_dir, image_name)

        process_image(input_path, output_dir)


def main() -> None:
    """
    Main function to parse CLI arguments and process images accordingly.
    """
    parser = argparse.ArgumentParser(description="Process glyph.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw",
        help="Directory containing input images to process. Default is 'data/raw'. This option is ignored if --input-file is provided.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to a single input image to process. Overrides --input-dir if provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed images and glyphs. Default is 'data/processed'.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Maximum number of images to process from the input directory.",
    )

    args = parser.parse_args()

    if args.input_file:
        # Process a single image
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file not found: {args.input_file}")

        process_image(args.input_file, args.output_dir)

    elif args.input_dir:
        # Process images in a directory
        if not os.path.exists(args.input_dir):
            raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

        process_images_in_directory(args.input_dir, args.output_dir, args.max_images)


if __name__ == "__main__":
    main()
