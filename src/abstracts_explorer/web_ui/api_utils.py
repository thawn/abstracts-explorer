"""Shared helpers for Flask API routes in the web UI."""

import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, ParamSpec, Tuple, TypeVar, overload

from flask import Request, jsonify, request
from werkzeug.exceptions import BadRequest, UnsupportedMediaType

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


class ApiError(Exception):
    """
    Controlled API error with an HTTP status code.

    Parameters
    ----------
    message : str
        Error message returned to the client.
    status_code : int, optional
        HTTP status code to return (default is 500).
    """

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@overload
def api_error_handler(func: Callable[P, R], *, error_message: Optional[str] = None) -> Callable[P, R]: ...


@overload
def api_error_handler(
    func: None = None, *, error_message: Optional[str] = None
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def api_error_handler(func: Optional[Callable[P, R]] = None, *, error_message: Optional[str] = None) -> Any:
    """
    Catch and convert exceptions raised by an API route into JSON responses.

    Parameters
    ----------
    func : callable, optional
        The Flask view function to wrap. When omitted, the decorator is
        returned so it can be used with keyword arguments.
    error_message : str, optional
        Generic message returned for unexpected exceptions. When omitted,
        ``str(exception)`` is returned to the client.

    Returns
    -------
    callable
        Decorated view function, or a decorator when ``func`` is omitted.
    """

    def decorator(view: Callable[P, R]) -> Callable[P, R]:
        @wraps(view)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> Any:
            try:
                return view(*args, **kwargs)
            except ApiError as exc:
                if exc.status_code >= 500:
                    logger.error("API error in %s: %s", request.endpoint, exc.message, exc_info=True)
                return jsonify({"error": exc.message}), exc.status_code
            except BadRequest as exc:
                logger.warning("Invalid request body in %s: %s", request.endpoint, exc)
                return jsonify({"error": "Invalid JSON request body"}), 400
            except UnsupportedMediaType as exc:
                logger.warning("Unsupported media type in %s: %s", request.endpoint, exc)
                return jsonify({"error": "Content-Type must be application/json"}), 415
            except Exception as exc:
                logger.error("Error in %s endpoint: %s", request.endpoint, exc, exc_info=True)
                message = error_message if error_message is not None else str(exc)
                return jsonify({"error": message}), 500

        return wrapped

    if func is None:
        return decorator
    return decorator(func)


def get_json_object() -> Dict[str, Any]:
    """Return the request JSON body as a dict or raise a 400 ``ApiError``."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ApiError("Request body must be a JSON object", 400)
    return data


def coerce_limit(value: Any, default: int = 10, minimum: int = 1, maximum: int = 1000) -> int:
    """Coerce a client-supplied result limit into a safe integer range."""
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ApiError(f"Invalid limit: {value}", 400) from exc
    if parsed < minimum or parsed > maximum:
        raise ApiError(f"limit must be between {minimum} and {maximum}", 400)
    return parsed


def coerce_int_list(value: Any, name: str) -> List[int]:
    """Coerce a client-supplied value into a list of integers."""
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ApiError(f"{name} must be a list", 400)
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ApiError(f"{name} must contain integers", 400) from exc


def coerce_str_list(value: Any, name: str) -> List[str]:
    """Coerce a client-supplied value into a list of strings."""
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ApiError(f"{name} must be a list", 400)
    return [str(item) for item in value]


def get_int_param(request: Request, name: str, default: Optional[int] = None) -> Optional[int]:
    """
    Read an integer query parameter from a request.

    Parameters
    ----------
    request : flask.Request
        The incoming Flask request.
    name : str
        Query parameter name.
    default : int, optional
        Value returned when the parameter is absent or empty.

    Returns
    -------
    int or None
        Parsed integer value, or ``default`` when not provided.

    Raises
    ------
    ApiError
        If the parameter is present but cannot be converted to an integer.
    """

    value = request.args.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ApiError(f"Invalid {name} parameter: {exc}", 400) from exc


def get_conference_year_params(
    request: Request,
) -> Tuple[Optional[str], Optional[int]]:
    """
    Read the common ``conference`` and ``year`` query parameters.

    Parameters
    ----------
    request : flask.Request
        The incoming Flask request.

    Returns
    -------
    tuple
        ``(conference, year)`` where each value is ``None`` when the
        corresponding query parameter is absent or empty.
    """

    conference_param = request.args.get("conference")
    conference = conference_param if conference_param else None
    year = get_int_param(request, "year")
    return conference, year
