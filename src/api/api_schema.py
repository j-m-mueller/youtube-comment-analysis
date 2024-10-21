"""src.api.api_schema.py -- schemata/data models for FastAPI endpoints"""

from pydantic import BaseModel
from typing import Optional


class AnalysisParams(BaseModel):
    """Parameters for Comment Analysis.

    :param analyze_donations: analyze YT donations.
    :param target_currency: convert donations to this currency.
    :param translate_comments: translate the comments to the target language (typically English) - highly recommended when dealing with non-english comments!
    :param translate_language-code: language code of the target language for translation.
    :param dislike_polarity_cutoff: comment polarity minimum to assume a highly positive/negative comment.
    """
    
    analyze_donations: Optional[bool] = True
    target_currency: Optional[str] = 'USD'
    translate_comments: Optional[bool] = True
    translate_language_code: Optional[str] = 'en'
    dislike_polarity_cutoff: Optional[float] = 0.5


class AnalysisRequest(BaseModel):
    """Schema for Comment Analysis Request submission."""
    
    raw_html: str
    params: Optional[AnalysisParams]


class CommentMetrics(BaseModel):
    raw_comments__count: int
    translated_comments__count: int
    sentiment_comments__vader__median: float
    sentiment_comments__textblob__median: float
    

class DonationMetrics(BaseModel):
    donations__count: int
    donations__sum: float
    donations__currency: str
    

class DislikeMetrics(BaseModel):
    highly_positive_comments__count: int
    highly_negative_comments__count: int
    likes__count: int
    dislikes__estimate: int


class AnalysisResponse(BaseModel):
    """Response to a Comment Analysis Request."""
    
    comments: CommentMetrics
    donations: Optional[DonationMetrics] = None
    dislikes: DislikeMetrics
