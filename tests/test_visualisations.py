"""tests.test_comment_processor.py -- tests for the FastAPI."""

import pathlib

from src.comment_analysis.comment_processor import CommentProcessor
from src.comment_analysis.visualisations import plot_results


CURRENT_DIR = pathlib.Path(__file__).parent


def test_visualisation_generation():
    """Test whether a valid input HTML yields visualisations."""
    
    cp = CommentProcessor()

    file_path = CURRENT_DIR / "sample_data" / "sample_data_complete.txt"
    
    with open(file_path, 'r', encoding='utf-8') as file:
        sample_data = file.read()

    response_dict = cp.process_data(raw_html=sample_data)
    
    plot_results(response_dict=response_dict,
                 hide_output=True)
