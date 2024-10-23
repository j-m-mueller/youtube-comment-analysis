"""tests.test_comment_processor.py -- tests for the FastAPI."""

import pytest
import requests

from conftest import API_PORT, API_HOST


def test_api_connection_available():
    """Test if the API was properly started."""
    try:
        response = requests.get(f'http://{API_HOST}:{API_PORT}/health')
        # Assert that the response was successful
        assert response.status_code == 200, f"status_code for accessing /health endpoint was {response.status_code}"
    except requests.ConnectionError as e:
        pytest.fail(f"ConnectionError occurred: {e}")


class TestAnalyzeCommentsEndpoint:
    """Tests for analyze_comments endpoint."""

    def test_analyze_comments_invalid_args(self):
        """Test if invalid params raise a ValidationError."""
        args = {
            "not_a_valid_param": "test",
            "raw_html": "<insert HTML here>",
            "params": {}
        }

        response = requests.post(url=f'http://{API_HOST}:{API_PORT}/analyze-comments',
                                 json=args)

        assert response.status_code == 422, f"status code for malformatted request is {response.status_code}, i.e., != 422"
        assert "not_a_valid_param" in str(response.json())

    def test_analyze_comments_invalid_html(self):
        """Test if invalid HTML leads to a from src.comment_analysis.exceptions import NoCommentsFoundException"""
        args = {
            "raw_html": "<insert HTML here>",
            "params": {}
        }

        response = requests.post(url=f'http://{API_HOST}:{API_PORT}/analyze-comments',
                                 json=args)

        assert response.status_code == 422, "status code for insufficient HTML: {response.status_code} (!= 422)"
        assert response.json() == "No/insufficient raw HTML provided."
