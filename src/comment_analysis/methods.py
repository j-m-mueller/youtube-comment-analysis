"""methods.py -- methods for donation and sentiment analysis."""

import nltk
import pandas as pd
import string

from currency_converter import CurrencyConverter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from typing import List, Dict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Donation Analysis

# dictionary for converting symbols to codes - extend if necessary
symbol_to_code_dictionary = {
    '€': 'EUR',
    '$': 'USD',
    '£': 'GBP',
    '¥': 'JPY',
    '₹': 'INR',
    '₩': 'KRW',
    '₦': 'NGN',
    '₽': 'RUB',
    'R$': 'BRL',
}


def get_currency_conversion_df(unique_currencies: List[str],
                               target_currency: str = 'USD') -> pd.DataFrame:
    """
    Get a DataFrame with conversion rates for the currencies of the donations on a video.

    :param unique_currencies: list of unique currencies.
    :param target_currency: currency to convert to (default: USD).

    :return: DataFrame with conversion rates.
    """
    currency_rates = CurrencyConverter()

    conv_rates = []
    for currency in unique_currencies:
        try:
            conv_rates.append(
                {
                    'currency': currency,
                    'factor': currency_rates.convert(
                        amount=1,
                        currency=symbol_to_code_dictionary.get(currency, currency), 
                        new_currency=target_currency
                    )
                }
            )
        except ValueError as e:
            raise ValueError(f"Invalid currency provided: {currency}")

    return pd.DataFrame(conv_rates)    


# Comment Concatenation

def concatenate_comments(comments: List[str],
                         separator: str = " ||| ", 
                         max_length: int = 3000) -> List[str]:
    """
    Concatenates a list of comments into larger strings, each with a maximum length limit. 
    When the concatenated string exceeds the max_length, the last comment is removed, 
    and a new concatenation starts.

    :param comments: List of text comments to be concatenated.
    :param separator: A string used to separate comments in the concatenated string. Default is ' ||| '.
    :param max_length: The maximum allowed length for each concatenated string. Default is 3000 characters. 
        Note: if higher numbers are chosen, translation requests may fail.
    :return: A list of concatenated strings where each string's length does not exceed the max_length.
    """
    concatenated_strings = []
    current_concat = ""

    for comment in comments:
        # add the separator only if current_concat is not empty
        next_concat = current_concat + (separator if current_concat else "") + comment
        
        # check if adding the next comment would exceed the max_length
        if len(next_concat) > max_length:
            # append the current concatenated string (without the new comment)
            concatenated_strings.append(current_concat)
            # start a new concatenated string with the current comment
            current_concat = comment
        else:
            # update the current concatenated string
            current_concat = next_concat

    # append the last concatenated string if it's not empty
    if current_concat:
        concatenated_strings.append(current_concat)

    return concatenated_strings


# Polarity Calculation

def get_comment_polarity_df(comments: List[str]) -> pd.DataFrame:
    """
    Calculate a polarity DF for a list of comments.

    :param comments: list of comments.

    :return: DataFrame with comment column and different polarity estimates including overall polarity (negative/neutral/positive).
    """
    vader_analyzer = SentimentIntensityAnalyzer()

    comment_df = pd.DataFrame(
        {
            'comment': comments,
            'polarity_textblob': [TextBlob(comment).sentiment.polarity for comment in comments],
            'polarity_vader': [vader_analyzer.polarity_scores(comment)['compound'] for comment in comments]
        }
    )

    for suffix in ['textblob', 'vader']:
        comment_df[f'sentiment_{suffix}'] = comment_df[f'polarity_{suffix}'].apply(
            lambda pol: 'positive' if pol > 0 else 'negative' if pol < 0 else 'neutral'
        )

    return comment_df
    

"""Term Extraction"""

# download and load stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def preprocess_text(text: str,
                    stop_words: List[str]) -> str:
    """
    Preprocess the text by stopword elimination, removal of punctuation, and tokenzation.

    :param text: input comment.
    :param stop_words: list of stop words to eliminate.

    :return: preprocessed string.
    """
    # lowercase
    text = text.lower()
    
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # tokenize and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)
        

def extract_relevant_terms(comment_df: pd.DataFrame,
                           polarity_column: str = 'polarity_vader', 
                           polarity_threshold: float = 0.5, 
                           max_features: int = 100) -> Dict[str, pd.Series]:
    """
    Extracts relevant terms from positive and negative comments based on TF-IDF scores.

    :param comment_df: input DF with comments and polarity.
    :param polarity_column: column in which the polarity is stored.
    :param polarity_threshold: threshold for the polarity score to apply (defaults to 0.5).
    :param max_features: number of terms to keep based on TF-IDF scores (defaults to 100).

    :return: returns TF-IDF terms in a pd.Series for both positive and negative comments.
    """
    # filter out highly positive and highly negative comments
    positive_comments = comment_df[comment_df[polarity_column] > polarity_threshold]['comment']
    negative_comments = comment_df[comment_df[polarity_column] < -polarity_threshold]['comment']

    # apply preprocessing to comments
    positive_comments = positive_comments.apply(lambda comment: preprocess_text(comment,
                                                                                stop_words=stop_words))
    negative_comments = negative_comments.apply(lambda comment: preprocess_text(comment,
                                                                                stop_words=stop_words))

    # combine the two sets into one DataFrame for TF-IDF
    combined_df = pd.concat([positive_comments, negative_comments], axis=0)

    # define a binary label to keep track of positive (1) and negative (0) comments
    labels = [1] * len(positive_comments) + [0] * len(negative_comments)

    # initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features,
                                 ngram_range=(1, 2))  # You can adjust max_features or n-grams

    # fit and transform the comments
    X = vectorizer.fit_transform(combined_df)

    # create a DataFrame from the TF-IDF matrix
    tfidf_df = pd.DataFrame(
        X.toarray(), 
        columns=vectorizer.get_feature_names_out()
    )

    # separate TF-IDF scores for positive and negative comments
    tfidf_positive = tfidf_df.iloc[:len(positive_comments)].mean().sort_values(ascending=False)
    tfidf_negative = tfidf_df.iloc[len(positive_comments):].mean().sort_values(ascending=False)

    # return the top terms for positive and negative comments
    return {
        'positive': tfidf_positive,
        'negative': tfidf_negative
    }
