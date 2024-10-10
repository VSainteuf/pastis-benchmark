import numpy as np


def masks_to_str(predictions: np.ndarray) -> list[str]:
    """
    Convert the

    Args:
        predictions (np.ndarray): predictions as a 3D batch (B, H, W)

    Returns:
        list[str]: a list of B strings, each string is a flattened stringified prediction mask
    """
    return [" ".join(f"{x}" for x in np.ravel(x)) for x in predictions]


def decode_masks(
    masks: list[str],
    target_shape: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """
    Convert each string in masks back to a 1D list of integers.

    Args:
        masks (list[str]): list of stringified masks

    Returns:
        np.ndarray: reconstructed batch of masks
    """
    return np.array(
        [
            np.fromstring(mask, sep=" ", dtype=np.uint8).reshape(target_shape)
            for mask in masks
        ]
    )
