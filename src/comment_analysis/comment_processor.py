"""src.comment_processor.py -- processor class for analysing YouTube comments."""

import logging
import pandas as pd
import re
import seaborn as sns

from bs4 import BeautifulSoup
from dataclasses import dataclass
from googletrans import Translator
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import List, Dict

from src.comment_analysis.methods import extract_relevant_terms, get_comment_polarity_df, concatenate_comments, get_currency_conversion_df
from src.comment_analysis.exceptions import NoCommentsFoundException, NoLikesFoundException, NoDonationsFoundException


logger = logging.getLogger()


@dataclass
class CommentProcessor:
    """Class for analysing YouTube comments."""
    
    analyze_donations: bool = True
    target_currency: str = 'USD'
    translate_comments: bool = True
    translate_language_code: str = 'en'
    dislike_polarity_cutoff: float = 0.5

    def process_comments(self,
                         raw_html: str) -> dict:
        """
        Process YouTube comments based on a provided HTML fragment.

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
            'metrics': self._estimate_dislikes(
                soup=soup,
                comment_df=response_dict['comments']['details']['comment_df']
            )
        }

        # analyze donations
        if self.analyze_donations:
            try:
                response_dict['donations'] = self._analyze_donations(soup=soup)
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
        translated_comments = self._translate_comments(comments=comments)
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

    def _translate_comments(self,
                            comments: List[str]) -> pd.DataFrame:
        """
        Get translations of raw comments.

        :param comments: list of comments.

        :return: DF with translated comment information.
        """
        separator: str = " ||| "

        # concatenate comments into larger chunks for submission to translation API
        concatenates = concatenate_comments(comments=comments,
                                            separator=separator)

        # translate chunks
        translator = Translator()
        
        raw_translations = []
        failed = []
        for i, concatenate in tqdm(
            enumerate(concatenates), 
            total=len(concatenates), 
            desc='Translating comment concatenates'
        ):
            try:
                raw_translations.append(translator.translate(concatenate, dest='en'))
            except Exception as e:
                logger.info(f"Error translating chunk {i}: {type(e)}: {e}")
                failed.append(concatenate)

        if len(failed) > 0:
            logger.info(f"Translation of {len(failed)} chunks failed.")

        # split chunks back into individual comments
        combined_chunks = separator.join([translation.text for translation in raw_translations])
        return combined_chunks.split("|||")

    def _estimate_dislikes(self,
                           soup: BeautifulSoup,
                           comment_df: pd.DataFrame) -> Dict[str, float]:
        """
        Dislike estimate based on extremely positive/negative comments and like count.

        :param soup: parsed HTML data.
        :param comment_df: DF with comment-related polarity.

        :return: dictionary with like/dislike metrics.
        """
        # get like count
        like_divs = soup.findAll('div', {'class': 'yt-spec-button-shape-next__button-text-content'})

        if len(like_divs) == 0:
            raise NoLikesFoundException("Like div not found! Please check provided HTLM data.")

        like_divs = [int(div.text.replace('.', '').strip()) for div in like_divs 
                     if re.match(r'^\d{1,3}(\.\d{3})*$', div.text.strip())  # match counts with decimal separator
                     or re.match(r'^\d+$', div.text.strip())]  # match counts without decimal separator
        like_count = like_divs[0]

        # get estimate of extremely positive/negative comment ratio
        comment_df['polarity_mean'] = (comment_df['polarity_vader'] + comment_df['polarity_textblob'])/ 2
        
        highly_positive_count = (comment_df['polarity_mean'] > self.dislike_polarity_cutoff).astype(int).sum()
        highly_negative_count = (comment_df['polarity_mean'] < -self.dislike_polarity_cutoff).astype(int).sum()
        comment_ratio = highly_positive_count / highly_negative_count

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
        
    def _analyze_donations(self,
                           soup: BeautifulSoup) -> dict:
        """
        Analyze donations embedded in YT comments.

        :param soup: parsed HTML data.

        :return: pd.DataFrame with donation data.
        """
        donation_df = self._get_donations_from_html(soup=soup)

        if len(donation_df) == 0:
            raise NoDonationsFoundException

        unique_currencies = donation_df['currency'].unique()
        
        curr_conv_df = get_currency_conversion_df(unique_currencies=unique_currencies,
                                                  target_currency=self.target_currency)
        
        merged_donation_df = donation_df.merge(curr_conv_df, left_on='currency', right_on='currency', how='left')
        merged_donation_df['conv_donation'] = merged_donation_df['donation'] * merged_donation_df['factor']

        logger.info(f"Total sum of donations: {merged_donation_df['conv_donation'].sum():.2f} {self.target_currency}\n")

        return {
            'details': {
                'donation_df': merged_donation_df
            },
            'metrics': {
                'donations__count': len(merged_donation_df),
                'donations__sum': round(merged_donation_df['conv_donation'].sum(), 2),
                'donations__currency': self.target_currency
            }
        }
    
    def _get_donations_from_html(self,
                                 soup: BeautifulSoup) -> pd.DataFrame:
        """
        Isolate donations from bs4 object.

        :param soup: BeautifulSoup object.

        :return: pd.DataFrame with donation data.
        """
        price_spans = soup.findAll('span', {'id': 'comment-chip-price'})
        donations = [elem.text.strip() for elem in price_spans if len(elem.text.strip()) > 0]
        donation_df = pd.DataFrame(
            [elem.split("\xa0") for elem in donations], 
            columns=['donation', 'currency']
        )
        donation_df['donation'] = donation_df['donation'].apply(
            lambda donation: donation.replace(',', '.')
        ).astype(float)

        return donation_df
