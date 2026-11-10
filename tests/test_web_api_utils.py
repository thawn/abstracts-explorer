"""Tests for web_ui/api_utils.py."""

import logging

import pytest
from flask import Flask, jsonify, request

from abstracts_explorer.web_ui.api_utils import (
    ApiError,
    api_error_handler,
    get_conference_year_params,
    get_int_param,
)


@pytest.fixture
def app():
    """Create a Flask app with routes exercising the shared API helpers."""
    test_app = Flask(__name__)

    @test_app.route("/ok")
    @api_error_handler
    def ok():
        return jsonify({"ok": True})

    @test_app.route("/controlled")
    @api_error_handler
    def controlled():
        raise ApiError("invalid value", 400)

    @test_app.route("/controlled-server")
    @api_error_handler
    def controlled_server():
        raise ApiError("storage unavailable")

    @test_app.route("/boom")
    @api_error_handler
    def boom():
        raise RuntimeError("kaboom")

    @test_app.route("/hidden")
    @api_error_handler(error_message="Something failed")
    def hidden():
        raise RuntimeError("secret detail")

    @test_app.route("/int")
    @api_error_handler
    def int_route():
        return jsonify({"value": get_int_param(request, "value")})

    @test_app.route("/int-default")
    @api_error_handler
    def int_default_route():
        return jsonify({"value": get_int_param(request, "value", 7)})

    @test_app.route("/conference-year")
    @api_error_handler
    def conference_year_route():
        conference, year = get_conference_year_params(request)
        return jsonify({"conference": conference, "year": year})

    return test_app


@pytest.fixture
def client(app):
    """Create a test client for the helper test app."""
    return app.test_client()


class TestGetIntParam:
    """Tests for get_int_param."""

    def test_returns_parsed_int(self, app):
        with app.test_request_context("/?value=42"):
            assert get_int_param(request, "value") == 42

    def test_missing_returns_none_by_default(self, app):
        with app.test_request_context("/"):
            assert get_int_param(request, "value") is None

    def test_missing_returns_custom_default(self, app):
        with app.test_request_context("/"):
            assert get_int_param(request, "value", 7) == 7

    def test_empty_returns_default(self, app):
        with app.test_request_context("/?value="):
            assert get_int_param(request, "value", 7) == 7

    def test_invalid_raises_api_error(self, app):
        with app.test_request_context("/?year=not-a-year"):
            with pytest.raises(ApiError) as exc_info:
                get_int_param(request, "year")
        assert exc_info.value.status_code == 400
        assert "Invalid year parameter" in exc_info.value.message


class TestGetConferenceYearParams:
    """Tests for get_conference_year_params."""

    def test_returns_conference_and_year(self, app):
        with app.test_request_context("/?conference=NeurIPS&year=2025"):
            assert get_conference_year_params(request) == ("NeurIPS", 2025)

    def test_missing_values_are_none(self, app):
        with app.test_request_context("/"):
            assert get_conference_year_params(request) == (None, None)

    def test_empty_values_are_none(self, app):
        with app.test_request_context("/?conference=&year="):
            assert get_conference_year_params(request) == (None, None)

    def test_only_conference(self, app):
        with app.test_request_context("/?conference=ICLR"):
            assert get_conference_year_params(request) == ("ICLR", None)

    def test_invalid_year_raises_api_error(self, app):
        with app.test_request_context("/?year=invalid"):
            with pytest.raises(ApiError) as exc_info:
                get_conference_year_params(request)
        assert exc_info.value.status_code == 400


class TestApiErrorHandler:
    """Tests for the api_error_handler decorator."""

    def test_successful_response_passes_through(self, client):
        response = client.get("/ok")
        assert response.status_code == 200
        assert response.get_json() == {"ok": True}

    def test_api_error_returns_status_and_message(self, client):
        response = client.get("/controlled")
        assert response.status_code == 400
        assert response.get_json() == {"error": "invalid value"}

    def test_api_error_defaults_to_500(self, client):
        response = client.get("/controlled-server")
        assert response.status_code == 500
        assert response.get_json() == {"error": "storage unavailable"}

    def test_unexpected_exception_returns_500(self, client):
        response = client.get("/boom")
        assert response.status_code == 500
        assert response.get_json() == {"error": "kaboom"}

    def test_custom_error_message_hides_details(self, client):
        response = client.get("/hidden")
        assert response.status_code == 500
        assert response.get_json() == {"error": "Something failed"}

    def test_unexpected_exception_is_logged(self, client, caplog):
        with caplog.at_level(logging.ERROR):
            client.get("/boom")
        assert "Error in boom endpoint" in caplog.text
        assert "kaboom" in caplog.text

    def test_client_error_is_not_logged_as_error(self, client, caplog):
        with caplog.at_level(logging.ERROR):
            client.get("/controlled")
        assert "invalid value" not in caplog.text

    def test_param_error_returns_json_400(self, client):
        response = client.get("/int?value=abc")
        assert response.status_code == 400
        assert "Invalid value parameter" in response.get_json()["error"]

    def test_param_helper_route_success(self, client):
        response = client.get("/int?value=12")
        assert response.get_json() == {"value": 12}

    def test_param_helper_route_default(self, client):
        response = client.get("/int-default")
        assert response.get_json() == {"value": 7}

    def test_conference_year_route(self, client):
        response = client.get("/conference-year?conference=ICML&year=2024")
        assert response.get_json() == {"conference": "ICML", "year": 2024}


def test_api_error_default_status():
    """ApiError defaults to HTTP 500."""
    error = ApiError("failure")
    assert error.message == "failure"
    assert error.status_code == 500
