import numpy as np
import matplotlib.pyplot as plt


def overlay_mask_on_image(color_image, mask, mask_color, alpha, offset_r=0, offset_c=0):
    """
    Overlay a black-and-white mask on a color image with specified color and alpha.

    Args:
        color_image (numpy.ndarray): HxWx3 array representing the color image (RGB).
        mask (numpy.ndarray): HxW binary array (0 or 1) representing the mask.
        mask_color (str or tuple): The desired color for the mask, e.g., 'red' or (1, 0, 0).
        alpha (float): Transparency of the mask overlay (0.0 to 1.0).
        offset_r (int): Row offset for the mask.
        offset_c (int): Column offset for the mask.

    Returns:
        numpy.ndarray: Combined image with the mask overlay applied.
    """
    # Ensure color image is in float format
    if color_image.max() > 1.0:
        color_image = color_image / 255.0  # Convert to float (0-1 range)

    # Convert grayscale to RGB if needed
    if color_image.ndim == 2:  # Grayscale image
        color_image = np.stack([color_image] * 3, axis=-1)  # Convert to HxWx3

    # Ensure mask is binary (0 or 1)
    mask = (mask > 0).astype(bool)

    # Convert color string to RGB if needed
    if isinstance(mask_color, str):
        import matplotlib.colors as mcolors

        mask_color = mcolors.to_rgb(mask_color)  # Convert to (R, G, B) tuple

    # Create an overlay of the mask color
    overlay = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    for i in range(3):  # Apply mask color to each channel
        overlay[:, :, i] = mask * mask_color[i]

    # Apply the offset to the mask
    offset_overlay = np.zeros_like(color_image, dtype=np.float32)
    offset_overlay[
        offset_r : offset_r + mask.shape[0], offset_c : offset_c + mask.shape[1]
    ] = overlay

    # Blend the color image with the overlay
    combined_image = np.where(
        mask[..., None],  # Apply mask_color with alpha where mask is True
        (1 - alpha) * color_image + alpha * offset_overlay,
        color_image,
    )
    return combined_image