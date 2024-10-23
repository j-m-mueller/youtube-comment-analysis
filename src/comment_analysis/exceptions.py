"""src.comment_analysis.exceptions.py -- exceptions raised throughout comment processing"""


class NoCommentsFoundException(Exception):
    """Exception raised if no comments are detected within a raw HTML document."""
    pass


class NoLikesFoundException(Exception):
    """Exception raised if no like count is detected within a raw HTML document."""
    pass


class NoDonationsFoundException(Exception):
    """Exception raised if no donations are detected within a raw HTML document."""
    pass
