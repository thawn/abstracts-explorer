"""
JSON serialization helpers shared across the package.
"""

import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that converts NumPy scalars and arrays to built-in types.

    Examples
    --------
    >>> import numpy as np
    >>> json.dumps({"n": np.int64(3), "x": np.float32(0.5)}, cls=NumpyEncoder)
    '{"n": 3, "x": 0.5}'
    """

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)
