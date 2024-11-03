"""src.comment_analysis.dislikes.py -- module for dislike processing."""

import logging
import pandas as pd
import re

from bs4 import BeautifulSoup
from typing import List

from src.comment_analysis.exceptions import NoLikesFoundException


logger = logging.getLogger(__name__)


class DislikeEstimator:
    """Class to extract dislikes from raw HTML."""
    
    def __init__(self,
                 dislike_polarity_cutoff: float = 0.5):
        """
        Initialize the class.
        
        :param dislike_polarity_cutoff: cutoff for the polarity to extract extremely positive/negative comments
            for dislike estimation (defaults to 0.5).
        """
        self._dislike_polarity_cutoff = dislike_polarity_cutoff
    
    def estimate_dislikes(self, 
                          soup: BeautifulSoup,
                          comment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Method to estimate dislikes based on parsed HTML data.
        
        :param soup: parsed HTML data.
        :param comment_df: DataFrame with comments with polarity estimates.
        
        :return: dictionary with like-related metrics and dislike estimate.
        """
        like_count = self._get_like_count(soup=soup)

        # get estimate of extremely positive/negative comment ratio
        comment_df['polarity_mean'] = (comment_df['polarity_vader'] + comment_df['polarity_textblob'])/ 2
        
        highly_positive_count = (comment_df['polarity_mean'] > self._dislike_polarity_cutoff).astype(int).sum()
        highly_negative_count = (comment_df['polarity_mean'] < -self._dislike_polarity_cutoff).astype(int).sum()
        comment_ratio = highly_positive_count / highly_negative_count

        # estimate dislike based on comment ratio
        dislike_estimate = int(like_count / comment_ratio)
        
        logger.info(f"Highly positive comments: {highly_positive_count}, highly negative comments: {highly_negative_count}, ratio positive/negative: {comment_ratio:.2f}")
        logger.info(f"Video likes: {like_count}")
        logger.info(f"Estimate of video dislikes based on likes and comment polarity: {dislike_estimate}\n")
        
        return {
            'highly_positive_comments__count': highly_positive_count,
            'highly_negative_comments__count': highly_negative_count,
            'positive_to_negative_comment_ratio': comment_ratio,
            'likes__count': like_count,
            'dislikes__estimate': dislike_estimate
        }
        
    def _get_like_count(self,
                        soup: BeautifulSoup) -> int:
        """
        Extract like count from parsed HTML data.
        
        :param soup: parsed HTML data.
        
        :return: like count  
        """
        # get like count
        like_divs = soup.findAll('div', {'class': 'yt-spec-button-shape-next__button-text-content'})

        if len(like_divs) == 0:
            raise NoLikesFoundException("Like div not found! Please check provided HTLM data.")

        like_divs = [int(div.text.replace('.', '').strip()) for div in like_divs 
                     if re.match(r'^\d{1,3}(\.\d{3})*$', div.text.strip())  # match counts with decimal separator
                     or re.match(r'^\d+$', div.text.strip())]  # match counts without decimal separator

        if len(like_divs) == 0:
            raise NoLikesFoundException("Like div not found! Please check provided HTLM data.")
        
        return like_divs[0]
    