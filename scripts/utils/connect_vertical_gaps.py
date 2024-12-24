from skimage import morphology, transform
import numpy as np

from utils.morphology import thinning_to_height


def connect_vertical_gaps(blank_image, char_height):
    rect_height = char_height // 3

    height, width = blank_image.shape[:2]
    resize_factor = 4
    if char_height < 60:
        resize_factor = 2
    if char_height < 30:
        resize_factor = 1

    samll_blank_image = transform.resize(
        blank_image,
        (height // resize_factor, width // resize_factor),
        anti_aliasing=False,
    )

    samll_blank_image = connect_vertical_gaps_morphology(
        samll_blank_image, rect_height // resize_factor
    )
    samll_blank_image = connect_vertical_gaps_morphology(
        samll_blank_image, rect_height // resize_factor
    )

    blank_image = transform.resize(
        samll_blank_image, (height, width), anti_aliasing=False
    )

    return blank_image


def connect_vertical_gaps_morphology(image, rect_height):
    closing_kernel = morphology.rectangle(rect_height, rect_height)
    image = morphology.binary_closing(image, closing_kernel)

    dilation_kernel = morphology.rectangle(rect_height * 15, rect_height)
    image = morphology.binary_dilation(image, dilation_kernel)

    closing_kernel = morphology.rectangle(rect_height * 15, 1)
    image = morphology.binary_closing(image, closing_kernel)

    closing_kernel = morphology.rectangle(rect_height * 15, 1)
    image = morphology.binary_closing(image, closing_kernel)

    opening_kernel = morphology.rectangle(2 * rect_height, 2 * rect_height)
    image = morphology.binary_opening(image, opening_kernel)

    return image
