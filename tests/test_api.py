"""tests.test_comment_processor.py -- tests for the FastAPI."""

import pathlib
import pytest
import requests

from conftest import API_PORT, API_HOST


CURRENT_DIR = pathlib.Path(__file__).parent


def test_api_connection_available():
    """Test if the API was properly started."""
    try:
        response = requests.get(f'http://{API_HOST}:{API_PORT}/health')
        # Assert that the response was successful
        assert response.status_code == 200, f"status_code for accessing /health endpoint was {response.status_code}"
    except requests.ConnectionError as e:
        pytest.fail(f"ConnectionError occurred: {e}")


def test_extract_comments_endpoint():
    """Test if the extract comments endpoint yields the expected number of comments."""
    
    file_path = CURRENT_DIR / "sample_data" / "sample_data_complete.txt"

    with open(file_path, 'r', encoding='utf-8') as file:
        sample_data = file.read()
    
    response = requests.post(url=f'http://{API_HOST}:{API_PORT}/extract-comments',
                             json={'raw_html': sample_data})
    comments = response.json()['comments']
    
    assert response.status_code == 200, f"Comment extraction request failed with status code {response.status_code}"
    assert isinstance(comments, list), f"Returned comment list is not of type list, but of type {type(comments)}"
    assert len(comments) == 5, f"Extracted number of comments should be 5, but is {len(comments)}"
    

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
