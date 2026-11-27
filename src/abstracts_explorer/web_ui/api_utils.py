"""Shared helpers for Flask API routes in the web UI."""

import logging
import math
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, ParamSpec, Tuple, TypeVar, overload

from flask import Request, jsonify, request
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge, UnsupportedMediaType

from abstracts_explorer.utils.http_utils import (
    is_llm_connection_error,
    is_llm_rate_limit_error,
    is_llm_timeout,
)
from abstracts_explorer.utils.read_only import ReadOnlyError

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

_TIMEOUT_ERROR_MESSAGE = (
    "The AI service took too long to respond. Please try again. "
    "If this keeps happening, try a shorter question or fewer papers."
)
_UNAVAILABLE_ERROR_MESSAGE = "The AI service is temporarily unavailable. Please try again in a moment."
_GENERIC_ERROR_MESSAGE = "Something went wrong while processing your request. Please try again."


def _format_seconds(value: float) -> str:
    """Format a duration without trailing decimals."""
    return f"{value:.0f}s"


def _rate_limit_details() -> Tuple[float, int, float]:
    """
    Return the configured rate-limit cooldown, attempt budget, and wait time.

    Returns
    -------
    tuple
        ``(cooldown_seconds, attempts, remaining_seconds)``.  The remaining
        time comes from the shared cooldown and falls back to the configured
        cooldown when no cooldown is active.
    """
    from abstracts_explorer.config import get_config
    from abstracts_explorer.rate_limiter import get_global_cooldown

    config = get_config()
    cooldown = float(getattr(config, "llm_rate_limit_delay", 60.0))
    attempts = max(1, int(getattr(config, "llm_max_attempts", 1)))
    remaining = get_global_cooldown().remaining()
    if remaining <= 0:
        remaining = cooldown
    return cooldown, attempts, remaining


def _rate_limit_message() -> str:
    """Build a user-facing message describing backend rate limiting and backoff."""
    cooldown, attempts, remaining = _rate_limit_details()
    return (
        "The AI service is rate-limiting requests (HTTP 429). Requests were retried "
        f"with exponential backoff and a shared cooldown of up to {_format_seconds(cooldown)} "
        f"across at most {attempts} attempts. Please wait about {_format_seconds(remaining)} "
        "and try again."
    )


def _timeout_message() -> str:
    """Build a timeout message, including recent rate-limit backoff when relevant."""
    from abstracts_explorer.rate_limiter import get_last_rate_limit

    cooldown, attempts, _ = _rate_limit_details()
    last = get_last_rate_limit(max_age=max(60.0, cooldown * attempts))
    if last is None:
        return _TIMEOUT_ERROR_MESSAGE
    delay, _ = last
    return (
        f"{_TIMEOUT_ERROR_MESSAGE} The AI backend was rate-limiting requests just before "
        f"the timeout (HTTP 429, backoff {_format_seconds(delay)}, up to {attempts} attempts), "
        "which likely caused the delay."
    )


def _backend_error_response(
    exc: BaseException,
    error_message: Optional[str] = None,
) -> Tuple[Any, int]:
    """Map backend failures to user-friendly messages and HTTP status codes."""
    if is_llm_rate_limit_error(exc):
        _, _, remaining = _rate_limit_details()
        response = jsonify({"error": _rate_limit_message()})
        response.headers["Retry-After"] = str(max(1, math.ceil(remaining)))
        return response, 429
    if is_llm_timeout(exc):
        return jsonify({"error": _timeout_message()}), 504
    if is_llm_connection_error(exc):
        return jsonify({"error": _UNAVAILABLE_ERROR_MESSAGE}), 503
    return jsonify({"error": error_message or _GENERIC_ERROR_MESSAGE}), 500


class ApiError(Exception):
    """
    Controlled API error with an HTTP status code.

    Parameters
    ----------
    message : str
        Error message returned to the client.
    status_code : int, optional
        HTTP status code to return (default is 500).
    headers : dict, optional
        Additional response headers.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.headers = headers or {}


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
        Generic message returned for unexpected non-backend exceptions. When
        omitted, a safe generic message is returned; technical details stay in
        the server log.

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
                    if is_llm_timeout(exc) or is_llm_connection_error(exc) or is_llm_rate_limit_error(exc):
                        return _backend_error_response(exc)
                response = jsonify({"error": exc.message})
                response.status_code = exc.status_code
                for header, value in exc.headers.items():
                    response.headers[header] = value
                return response
            except ReadOnlyError as exc:
                logger.info("Write blocked in read-only mode in %s: %s", request.endpoint, exc)
                return jsonify({"error": str(exc)}), 403
            except BadRequest as exc:
                logger.warning("Invalid request body in %s: %s", request.endpoint, exc)
                return jsonify({"error": "Invalid JSON request body"}), 400
            except UnsupportedMediaType as exc:
                logger.warning("Unsupported media type in %s: %s", request.endpoint, exc)
                return jsonify({"error": "Content-Type must be application/json"}), 415
            except RequestEntityTooLarge as exc:
                logger.warning("Request body too large in %s: %s", request.endpoint, exc)
                return jsonify({"error": "Request body too large"}), 413
            except Exception as exc:
                logger.error("Error in %s endpoint: %s", request.endpoint, exc, exc_info=True)
                return _backend_error_response(exc, error_message)

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


def coerce_int_list(value: Any, name: str, max_items: int = 100) -> List[int]:
    """Coerce a client-supplied value into a list of integers."""
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ApiError(f"{name} must be a list", 400)
    if len(value) > max_items:
        raise ApiError(f"{name} must contain at most {max_items} items", 400)
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ApiError(f"{name} must contain integers", 400) from exc


def coerce_str_list(
    value: Any,
    name: str,
    max_items: int = 100,
    max_item_length: int = 200,
) -> List[str]:
    """Coerce a client-supplied value into a list of strings."""
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ApiError(f"{name} must be a list", 400)
    if len(value) > max_items:
        raise ApiError(f"{name} must contain at most {max_items} items", 400)
    result = [str(item) for item in value]
    if any(len(item) > max_item_length for item in result):
        raise ApiError(f"{name} items must be at most {max_item_length} characters", 400)
    return result


def coerce_text(value: Any, name: str, max_length: int, default: str = "") -> str:
    """Coerce a client-supplied value into a length-bounded string."""
    text = default if value is None else str(value)
    if len(text) > max_length:
        raise ApiError(f"{name} must be at most {max_length} characters", 400)
    return text


def coerce_number(
    value: Any,
    name: str,
    default: Optional[float] = None,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    """Coerce a client-supplied value into a finite, bounded number."""
    if value is None:
        if default is None:
            raise ApiError(f"{name} is required", 400)
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ApiError(f"{name} must be a number", 400) from exc
    if not math.isfinite(parsed):
        raise ApiError(f"{name} must be a finite number", 400)
    if minimum is not None and parsed < minimum:
        raise ApiError(f"{name} must be at least {minimum}", 400)
    if maximum is not None and parsed > maximum:
        raise ApiError(f"{name} must be at most {maximum}", 400)
    return parsed


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
