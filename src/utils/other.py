from pprint import pprint
from typing import Any, List, Union
from warnings import warn

import numpy as np

# Avoiding some train imports for inference deployment
try:
    import pytorch_lightning as pl
except ImportError:
    warn("Could not import pytorch_lightning. Make sure it is installed.")

# Avoiding some heavy imports for "torch-less" Series Selector usage
try:
    import torch
    import torch.nn as nn
    from monai.utils import set_determinism
except ImportError:
    warn(
        "Could not import torch or monai. "
        "This can happen when heavy dependencies are not installed. "
        "Some functionalities may be limited."
    )


def seed_everything(seed: int, seed_monai: int) -> None:
    deterministic = True if seed else False
    if deterministic:
        pl.seed_everything(seed=seed, workers=True)

    if seed_monai is not None:

        # import random
        # import numpy as np
        set_determinism(seed=seed_monai)
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(4)
        # random.seed(seed_monai)
        # np.random.seed(seed_monai)
        # torch.manual_seed(seed_monai)
        # torch.cuda.manual_seed_all(seed_monai)
        # torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True, warn_only=True)

    return deterministic


def init_activation(activation_type, dim=1):
    if activation_type == "softmax":
        return nn.Softmax(dim=dim)
    elif activation_type == "sigmoid":
        return nn.Sigmoid()
    elif activation_type == "argmax":
        return lambda x: torch.argmax(x, dim=dim)
    elif activation_type == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")


def init_activation_numpy(activation_type, dim=1):
    if activation_type == "argmax":
        return lambda x: np.argmax(x, axis=dim)
    elif activation_type == "identity":
        return lambda x: x
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")


def init_aggregation(aggregation_type, dim=0):
    # Across stack dimension and preserve the dimension
    if aggregation_type == "mean":
        return lambda x: torch.mean(x, dim=dim, keepdim=True)
    elif aggregation_type == "max":
        return lambda x: torch.max(x, dim=dim, keepdim=True)[0]
    else:
        raise ValueError(f"Invalid aggregation type: {aggregation_type}")


def log_or_print(
    logger,
    message: str,
    use_pprint: bool = False,
    logging_method: str = "info",
):
    """
    Log or print a message.

    Args:
        logger: Logger instance. If None, the message will be printed.
        message (str): The message to log or print.
        use_pprint (bool): Whether to use pprint for printing. Defaults to False.
        logging_method (str): Logging method to use when logger is provided. Defaults to "info".
            Supported methods: "info", "debug", "warning", "error", "critical".
    """
    if logger is None:
        if use_pprint:
            pprint(message)
        else:
            print(message)
    else:
        # Dynamically call the logging method using getattr
        log_func = getattr(logger, logging_method, None)
        if callable(log_func):
            log_func(message)
        else:
            raise ValueError(f"Unsupported logging method: {logging_method}")


def sort_by_multiple_keys(
    values: List[Any],
    *sorting_vars: List[List[float]],
    ascending: Union[bool, List[bool]] = True,
) -> List[Any]:
    """
    Sort values by multiple sorting keys with support for ascending/descending order.

    Args:
        values: List of values to sort.
        *sorting_vars: Lists of keys to sort by (highest priority first).
        ascending: Bool or list of bools indicating ascending (True) or descending (False) for each key.

    Returns:
        Sorted list of values.
    """
    n = len(values)
    if not all(len(k) == n for k in sorting_vars):
        raise ValueError(
            "All sorting variables must have the same length as values."
        )

    num_keys = len(sorting_vars)

    if isinstance(ascending, bool):
        ascending = [ascending] * num_keys
    if len(ascending) != num_keys:
        raise ValueError(
            "Length of ascending list must match number of sorting keys."
        )

    # Adjust keys based on ascending flags
    adjusted_keys = [
        [k * (1 if asc else -1) for k in key]
        for key, asc in zip(sorting_vars, ascending)
    ]

    # Zip values with adjusted keys and sort
    indexed = list(zip(values, *adjusted_keys))
    sorted_indexed = sorted(indexed, key=lambda x: x[1:])
    return [x[0] for x in sorted_indexed]


def slices_to_bbox_array(slices_list):
    """
    Convert a list of bounding boxes in slice format to a numpy array of [N, 6] with [x1, y1, z1, x2, y2, z2] format.

    Parameters:
        slices_list (list of tuples): Each tuple has three slice objects (z, y, x) or (x, y, z).

    Returns:
        np.ndarray: Array of shape [N, 6] with bounding boxes in [x1, y1, z1, x2, y2, z2] format.
    """
    bboxes = []
    for slices in slices_list:
        if not all(isinstance(s, slice) for s in slices) or len(slices) != 3:
            raise ValueError(
                "Each element must be a tuple of 3 slice objects."
            )

        x1, y1, z1 = slices[0].start, slices[1].start, slices[2].start
        x2, y2, z2 = slices[0].stop, slices[1].stop, slices[2].stop
        bboxes.append([x1, y1, z1, x2, y2, z2])

    return np.array(bboxes, dtype=np.int32)
