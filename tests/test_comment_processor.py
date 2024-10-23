"""tests.test_comment_processor.py -- tests for CommentProcessor class."""

import pathlib
import pytest

from src.comment_analysis.comment_processor import CommentProcessor
from src.comment_analysis.exceptions import NoCommentsFoundException, NoLikesFoundException


CURRENT_DIR = pathlib.Path(__file__).parent


def test_insufficient_raw_html_provided():
    """Test CommentProcessor output if no valid HTML is provided."""

    cp = CommentProcessor()

    with pytest.raises(NoCommentsFoundException):
        cp.process_comments(raw_html='No comments contained here')

def test_valid_html_with_no_likes():
    """Test CommentProcessor response on valid request lacking likes data."""
    
    cp = CommentProcessor()

    file_path = CURRENT_DIR / "sample_data" / "sample_data_no_likes.txt"

    with open(file_path, 'r', encoding='utf-8') as file:
        sample_data = file.read()

    with pytest.raises(NoLikesFoundException):
        cp.process_comments(raw_html=sample_data)

def test_valid_html_with_no_comments():
    """Test CommentProcessor response on valid request lacking comment data."""
    
    cp = CommentProcessor()

    file_path = CURRENT_DIR / "sample_data" / "sample_data_no_comments.txt"

    with open(file_path, 'r', encoding='utf-8') as file:
        sample_data = file.read()

    with pytest.raises(NoCommentsFoundException):
        cp.process_comments(raw_html=sample_data)

def test_valid_html_no_donations():
    """Test CommentProcessor response on valid request lacking donation data."""
    
    cp = CommentProcessor()

    file_path = CURRENT_DIR / "sample_data" / "sample_data_no_donations.txt"
    
    with open(file_path, 'r', encoding='utf-8') as file:
        sample_data = file.read()

    response = cp.process_comments(raw_html=sample_data)

    assert 'donations' not in response.keys(), 'donation metrics key found in response dictionary despite absence of donations'
    assert 'comments' in response.keys(), 'comment metrics not in response dictionary despite presence of comments in HTML document'
    assert 'dislikes' in response.keys(), 'dislike metrics not in response dictionary despite presence of dislikes in HTML document'

def test_valid_html_complete_data():
    """Test CommentProcessor response on valid input with comments, dislikes, and donations."""
    
    cp = CommentProcessor()

    file_path = CURRENT_DIR / "sample_data" / "sample_data_complete.txt"
    
    with open(file_path, 'r', encoding='utf-8') as file:
        sample_data = file.read()

    response = cp.process_comments(raw_html=sample_data)

    assert 'comments' in response.keys(), 'comment metrics not in response dictionary despite presence of comments in HTML document'
    assert 'dislikes' in response.keys(), 'dislike metrics not in response dictionary despite presence of dislikes in HTML document'
    assert 'donations' in response.keys(), 'donation metrics not in response dictionary despite presence of donations in HTML document'
