"""src.comment_processor.py -- processor class for analysing YouTube comments."""

import logging
import pandas as pd

from bs4 import BeautifulSoup
from dataclasses import dataclass
from googletrans import Translator
from tqdm import tqdm
from typing import List

from src.comment_analysis.nlp import extract_relevant_terms, get_comment_polarity_df
from src.comment_analysis.dislikes import DislikeEstimator
from src.comment_analysis.donations import DonationProcessor
from src.comment_analysis.exceptions import NoCommentsFoundException, NoDonationsFoundException
from src.comment_analysis.translation import CommentTranslator


logger = logging.getLogger(__name__)


@dataclass
class CommentProcessor:
    """Class for analysing YouTube comments."""
    
    analyze_donations: bool = True
    target_currency: str = 'USD'
    translate_comments: bool = True
    translate_language_code: str = 'en'
    dislike_polarity_cutoff: float = 0.5

    def process_data(self,
                     raw_html: str) -> dict:
        """
        Process YouTube data based on a provided HTML fragment.

        :param raw_html: HTLM body (raw).

        :return: dictionary with Pandas DataFrames of various analysis results and additional information.
        """
        response_dict = {}

        if raw_html == '<!-- paste HTML <body ...>...</body> here! -->':
            raise NoCommentsFoundException("Please provide a valid HTML body for analysis (locally in `html_body.txt` or via request JSON)!")
        
        soup = BeautifulSoup(raw_html, 'html.parser')

        # analyze comments
        response_dict['comments'] = self._analyze_comments(soup=soup)

        # estimate dislikes
        response_dict['dislikes'] = {
            'metrics': DislikeEstimator(
                dislike_polarity_cutoff=self.dislike_polarity_cutoff
            ).estimate_dislikes(
                soup=soup,
                comment_df=response_dict['comments']['details']['comment_df']
            )
        }

        # analyze donations
        if self.analyze_donations:
            try:
                response_dict['donations'] = DonationProcessor(
                    target_currency=self.target_currency
                ).process_donations(
                    soup=soup
                )
            except NoDonationsFoundException:
                logger.info("No donations identified in HTML document.")

        return response_dict

    def _analyze_comments(self,
                          soup: BeautifulSoup) -> dict:
        """
        Analyze comment data from parsed HTML.

        :param soup: parsed HTML data.

        :return: dictionary with comment-related data.
        """
        comments = self._get_comments_from_html(soup=soup)
        translated_comments = CommentTranslator().translate_comments(comments=comments)
        comment_df = get_comment_polarity_df(comments=translated_comments)
        relevant_terms = extract_relevant_terms(comment_df=comment_df)

        logger.info(f"Extracted {len(comments)} raw comments and successfully translated {len(translated_comments)} of them.")
        logger.info(f"Comment polarity median: VADER: {comment_df['polarity_vader'].median():.2f}, TextBlob: {comment_df['polarity_textblob'].median():.2f}\n")

        return {
            'metrics': {
                'raw_comments__count': len(comments),
                'translated_comments__count': len(translated_comments),
                'sentiment_comments__vader__median': round(comment_df['polarity_vader'].median(), 4),
                'sentiment_comments__textblob__median': 
round(comment_df['polarity_textblob'].median(), 4)
            },
            'details': {
                'raw_comments': comments,
                'comment_df': comment_df,
                'relevant_terms': relevant_terms
            }
        }

    def _get_comments_from_html(self, 
                                soup: BeautifulSoup) -> List[str]:
        """
        Isolate comments from bs4 object.

        :param soup: parsed HTML data.

        :return: list with comment texts.
        """
        content_divs = soup.find('div', {'id': 'contents', 
                                         'class': 'style-scope ytd-item-section-renderer style-scope ytd-item-section-renderer'})

        if content_divs is None:
            raise NoCommentsFoundException("No comments identified within raw HTML. Please check provided data!")
        
        comment_spans = content_divs.findAll('span', {'role': 'text'})
        comments = [comment.text.strip() for comment in comment_spans 
                    if all(term not in comment.text.strip() for term in ['Antwort', 'reply', 'replies'])]

        return comments
