import numpy as np


def create_offset_mask(mask: np.ndarray, offset: int) -> np.ndarray:
    """
    Creates a new mask by offsetting the input mask.
    Positive offsets add empty rows at the top, shifting the mask down.
    Negative offsets add empty rows at the bottom, shifting the mask up.

    Args:
        mask (np.ndarray): Original binary mask.
        offset (int): Number of rows to offset the mask. Positive offsets shift down, negative offsets shift up.

    Returns:
        np.ndarray: Offset binary mask.
    """
    if abs(offset) >= mask.shape[0]:
        raise ValueError(
            "Absolute value of offset must be less than the number of rows in the mask."
        )

    # Create an empty mask of the same shape
    offset_mask = np.zeros_like(mask, dtype=mask.dtype)

    if offset > 0:
        # Shift mask down by `offset` rows
        offset_mask[offset:] = mask[:-offset]
    elif offset < 0:
        # Shift mask up by `abs(offset)` rows
        offset_mask[:offset] = mask[-offset:]
    else:
        # No offset, copy the mask
        offset_mask = mask.copy()

    return offset_mask
