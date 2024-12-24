import numpy as np
import torch
from torchvision import transforms
from skimage.transform import resize

GLYPH_HIGHT = 32
GLYPH_WIDTH = 32


class CustomTransform:
    def __call__(self, image):
        # Validate input image
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None.")

        # Ensure image has at least 2 dimensions
        if image.ndim < 2:
            raise ValueError(
                "Input image must have at least 2 dimensions (height, width)."
            )

        original_height, original_width = image.shape[:2]

        # Resize height to GLYPH_HIGHT while maintaining aspect ratio
        aspect_ratio = original_width / original_height if original_height > 0 else 1
        new_width = max(int(GLYPH_HIGHT * aspect_ratio), 1)

        # Resize the image to have height GLYPH_HIGHT and appropriate width
        image = resize(image, (GLYPH_HIGHT, new_width), anti_aliasing=True)

        # Handle width adjustment
        if new_width > GLYPH_WIDTH:
            # Crop the width to GLYPH_WIDTH pixels (crop centrally)
            excess_width = new_width - GLYPH_WIDTH
            start = excess_width // 2
            image = image[:, start : start + GLYPH_WIDTH]
        else:
            # Pad the width to GLYPH_WIDTH pixels (pad centrally)
            pad_width = (GLYPH_WIDTH - new_width) // 2
            image = np.pad(
                image,
                (
                    (0, 0),  # No padding for height
                    (pad_width, GLYPH_WIDTH - new_width - pad_width),
                ),  # Pad width to GLYPH_WIDTH
                mode="constant",
                constant_values=0,
            )

        # Handle grayscale images by adding a channel dimension
        if image.ndim == 2:  # Grayscale image
            image = np.expand_dims(image, axis=-1)

        # Ensure image has 3 channels (repeat grayscale across channels)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        # Convert to torch tensor and normalize
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Normalize the image
        image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)

        return image
