"""src.comment_analysis.exceptions.py -- exceptions raised throughout comment processing"""


class NoCommentsFoundException(Exception):
    """Exception raised if no comments are detected within a raw HTML document."""
    pass
