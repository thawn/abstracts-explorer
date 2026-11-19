"""
Tests for JSON serialization helpers.
"""

import json

import numpy as np
import pytest

from abstracts_explorer.utils.json_utils import NumpyEncoder


class TestNumpyEncoder:
    """Tests for NumpyEncoder NumPy type conversion."""

    def test_integer_scalar(self):
        assert json.dumps(np.int64(3), cls=NumpyEncoder) == "3"

    def test_floating_scalar(self):
        assert json.loads(json.dumps(np.float32(0.5), cls=NumpyEncoder)) == pytest.approx(0.5)

    def test_array(self):
        assert json.loads(json.dumps(np.array([1, 2, 3]), cls=NumpyEncoder)) == [1, 2, 3]

    def test_nested_structure(self):
        payload = {"n": np.int64(1), "x": np.float64(2.5), "arr": np.array([1.0, 2.0])}
        decoded = json.loads(json.dumps(payload, cls=NumpyEncoder))
        assert decoded == {"n": 1, "x": 2.5, "arr": [1.0, 2.0]}

    def test_unsupported_type_raises_type_error(self):
        with pytest.raises(TypeError):
            json.dumps(object(), cls=NumpyEncoder)
