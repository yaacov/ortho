from skimage.io import imsave
import numpy as np
import torch


def save_inverted_image(image: np.ndarray, output_path: str) -> None:
    """
    Invert and save the cleaned image to the specified path.

    Args:
        image (np.ndarray): Input image array.
        output_path (str): Path to save the processed image.
    """
    # Ensure the input image is in the range [0, 255] and type uint8
    if image.dtype != np.uint8:
        image = (
            (image * 255).astype(np.uint8)
            if image.max() <= 1
            else image.astype(np.uint8)
        )

    # Convert the image to a PyTorch tensor for manipulation
    image_tensor = torch.tensor(image, dtype=torch.uint8)

    # Invert the image using PyTorch operations
    inverted_tensor = 255 - image_tensor

    # Convert the inverted tensor back to a NumPy array
    inverted_image = inverted_tensor.numpy()

    # Save the image using skimage
    imsave(output_path, inverted_image)
