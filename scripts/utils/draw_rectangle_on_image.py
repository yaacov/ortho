import numpy as np
import gc
from skimage.draw import rectangle_perimeter
from matplotlib import colors


def draw_rectangle_on_image(image, start, end, color="red"):
    """Draw rectangle outline on image.

    Args:
        image: numpy array of image
        start: (minr, minc) tuple of start coordinates
        end: (maxr, maxc) tuple of end coordinates
        color: color string name (e.g. 'red') or RGB values as list [r,g,b]

    Returns:
        numpy array with rectangle drawn on it
    """
    # Create copy of input image
    result = image.copy()

    # Convert color string to RGB using matplotlib
    rgb_color = colors.to_rgb(color) if isinstance(color, str) else color

    # Get rectangle coordinates
    rr, cc = rectangle_perimeter(start=start, end=end)
    rr = np.clip(rr, 0, result.shape[0] - 1)
    cc = np.clip(cc, 0, result.shape[1] - 1)

    # Draw rectangle
    result[rr, cc] = rgb_color

    # Cleanup temporary arrays
    del rr, cc
    gc.collect()

    return result
