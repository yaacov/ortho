from matplotlib import pyplot as plt
import numpy as np
import gc


def draw_text_on_image(image, text, x, y, color="yellow", fontsize=32, weight=None):
    """Draw text on image using matplotlib.

    Args:
        image: numpy array of image
        text: string to draw
        x: pixel x coordinate
        y: pixel y coordinate
        color: text color
        fontsize: text font size
        weight: text font weight

    Returns:
        numpy array with text drawn on it
    """
    # Convert pixel coordinates to normalized (0-1) range
    text_x = x / image.shape[1]
    text_y = 1 - (y / image.shape[0])  # Invert y-axis for matplotlib

    plt.clf()  # Clear current figure
    with plt.ioff():
        # Create figure with exact image dimensions
        dpi = 100
        height, width = image.shape[:2]
        fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi, frameon=False)
        
        try:
            # Draw image and text
            fig.figimage(image, resize=False)
            fig.text(text_x, text_y, text, color=color, fontsize=fontsize, weight=weight)
            
            # Force draw and get buffer
            fig.canvas.draw()
            buffer = fig.canvas.buffer_rgba()
            
            # Convert to array
            result = np.frombuffer(buffer, dtype=np.uint8).copy()
            result = result.reshape(height, width, 4)[:, :, :3]
            result = result.astype(np.float32) / 255
            
            return result
            
        finally:
            # Cleanup
            plt.close(fig)
            plt.close('all')
            gc.collect()
