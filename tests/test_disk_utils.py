"""
Tests for disk-space helpers.
"""

from pathlib import Path

import pytest

from abstracts_explorer.utils.disk_utils import (
    InsufficientDiskSpaceError,
    available_bytes,
    ensure_free_space,
)


class TestAvailableBytes:
    """Tests for :func:`available_bytes`."""

    def test_existing_directory(self, tmp_path):
        """An existing directory reports a positive free-space value."""
        assert available_bytes(tmp_path) > 0

    def test_nonexistent_path_uses_existing_parent(self, tmp_path):
        """A not-yet-created path falls back to the nearest existing parent."""
        target = tmp_path / "does" / "not" / "exist"
        assert available_bytes(target) > 0

    def test_accepts_string_path(self, tmp_path):
        """String paths are accepted."""
        assert available_bytes(str(tmp_path)) > 0


class TestEnsureFreeSpace:
    """Tests for :func:`ensure_free_space`."""

    def test_passes_when_enough_space(self, tmp_path):
        """No error when the requirement is below the available space."""
        free = ensure_free_space(tmp_path, 1)
        assert free > 0

    def test_non_positive_requirement_is_noop(self, tmp_path):
        """A non-positive requirement never raises."""
        assert ensure_free_space(tmp_path, 0) > 0
        assert ensure_free_space(tmp_path, -5) > 0

    def test_raises_when_space_is_insufficient(self, tmp_path):
        """An impossible requirement raises a descriptive error."""
        required = 10**18
        with pytest.raises(InsufficientDiskSpaceError) as excinfo:
            ensure_free_space(tmp_path, required, description="embedding store")

        message = str(excinfo.value)
        assert "embedding store" in message
        assert str(required) in message

    def test_error_is_oserror(self, tmp_path):
        """The error is an ``OSError`` so callers can catch it generically."""
        assert issubclass(InsufficientDiskSpaceError, OSError)

    def test_uses_nearest_existing_parent(self, tmp_path):
        """Requirement checking works for a not-yet-created store path."""
        target = Path(tmp_path) / "chroma_db"
        assert ensure_free_space(target, 1) > 0
