"""src.comment_analysis.translation.py -- module for comment translation."""

import logging
import pandas as pd

from googletrans import Translator
from typing import List
from tqdm import tqdm


logger = logging.getLogger(__name__)


class CommentTranslator:
    """Class for the translation of comments."""
    
    def __init__(self,
                 max_concat_length: int = 3000,
                 concat_separator: str = ' ||| ',
                 target_language_code: str = 'en'):
        """
        Initialize the translator.
        
        :param max_concat_length: maximum allowed length for each concatenated string (defaults to 3000 characters). 
            Note: if higher numbers are chosen, translation requests may fail.
        :param concat_separator: string used to separate comments in the concatenated string (defaults to ' ||| ').
        :param target_language_code: target language to which to translate (defaults to 'en' for English).
        """
        self._max_concat_length = max_concat_length
        self._concat_separator = concat_separator
        self._target_language_code = target_language_code
    
    def translate_comments(self,
                           comments: List[str]) -> List[str]:
        """
        Get translations of raw comments.

        :param comments: list of comments.

        :return: list with translated comments.
        """
        separator: str = " ||| "

        # concatenate comments into larger chunks for submission to translation API
        concatenates = self._concatenate_comments(comments=comments)

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
                raw_translations.append(translator.translate(concatenate, dest=self._target_language_code))
            except Exception as e:
                logger.info(f"Error translating chunk {i}: {type(e)}: {e}")
                failed.append(concatenate)

        if len(failed) > 0:
            logger.info(f"Translation of {len(failed)} chunks failed.")

        # split chunks back into individual comments
        combined_chunks = separator.join([translation.text for translation in raw_translations])
        return combined_chunks.split("|||")

    def _concatenate_comments(self,
                              comments: List[str]) -> List[str]:
        """
        Concatenate a list of comments into larger strings with a specified separator and a maximal length.

        :param comments: list of text comments to be concatenated.
            
        :return: list of concatenated strings where each string's length does not exceed the max_length.
        """
        concatenated_strings = []
        current_concat = ""

        for comment in comments:
            # add the separator only if current_concat is not empty
            next_concat = current_concat + (self._concat_separator if current_concat else "") + comment
            
            # if maximal length is exceeded, start a new concatenated string with the current comment
            if len(next_concat) > self._max_concat_length:
                concatenated_strings.append(current_concat)
                current_concat = comment
            # else, update the current string by adding the current comment
            else:
                current_concat = next_concat

        # append the last concatenated string if it's not empty
        if current_concat:
            concatenated_strings.append(current_concat)

        return concatenated_strings