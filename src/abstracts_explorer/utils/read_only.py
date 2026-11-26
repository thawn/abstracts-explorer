"""
Read-only mode helpers.

Read-only mode prevents any write to the paper database or the embedding
store.  Write operations raise :class:`ReadOnlyError`, which the web UI maps
to HTTP 403.
"""

from functools import wraps
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class ReadOnlyError(RuntimeError):
    """Raised when a write is attempted while read-only mode is active."""


def require_writable(action: str = "") -> Callable[[F], F]:
    """
    Return a decorator that rejects write methods in read-only mode.

    Parameters
    ----------
    action : str, optional
        Human-readable description of the blocked operation.  Defaults to the
        decorated method name.

    Returns
    -------
    callable
        Decorator for a method whose instance exposes a ``read_only`` flag.
    """

    def decorator(method: F) -> F:
        @wraps(method)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if getattr(self, "read_only", False):
                label = action or method.__name__.replace("_", " ")
                raise ReadOnlyError(f"Cannot {label} in read-only mode")
            return method(self, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
