"""src.comment_analysis.nlp.py -- module for Natural Language Processing tasks."""

import nltk
import pandas as pd
import re
import string

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from typing import List, Dict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Polarity Calculation (Sentiment Analysis)

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
    

# Term Extraction (TF-IDF)

# download and load stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def preprocess_text(text: str,
                    stop_words: List[str],
                    remove_numbers: bool = True) -> str:
    """
    Preprocess the text by stopword elimination, removal of punctuation, and tokenzation.

    :param text: input comment.
    :param stop_words: list of stop words to eliminate.
    :param remove_numbers: boolean trigger to specify if numbers should be removed from the string.

    :return: preprocessed string.
    """
    # lowercase
    text = text.lower()
    
    # remove punctuation
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    
    # remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # tokenize and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)
        

def extract_relevant_terms(comment_df: pd.DataFrame,
                           polarity_column: str = 'polarity_vader', 
                           polarity_threshold: float = 0.5, 
                           max_features: int = 1000,
                           analyze_n_grams: bool = False) -> Dict[str, pd.Series]:
    """
    Extracts relevant terms from positive and negative comments based on TF-IDF scores.

    :param comment_df: input DF with comments and polarity.
    :param polarity_column: column in which the polarity is stored.
    :param polarity_threshold: threshold for the polarity score to apply (defaults to 0.5).
    :param max_features: number of terms to keep based on TF-IDF scores (defaults to 1000).
    :param analyze_n_grams: analyze n-grams in addition to single terms.

    :return: returns TF-IDF terms in a pd.Series for both positive and negative comments.
    """
    # preprocess comment data
    comment_df = comment_df.copy()
    comment_df['comment'] = comment_df['comment'].apply(lambda comment: preprocess_text(text=comment,
                                                                                        stop_words=stop_words))

    # fit vectorizer on whole corpus and transform into TF-IDF matrix
    vectorizer = TfidfVectorizer(max_features=max_features,
                                 ngram_range=(1, 2) if analyze_n_grams else (1, 1))
    tfidf_matrix = vectorizer.fit_transform(comment_df['comment'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    
    # extract TF-IDF scores for terms within extremely positive / negative comments and sort in descending order
    terms = {}
    for polarity_name, query in zip(['positive', 'negative'],
                                    [f"{polarity_column} > {polarity_threshold}", f"{polarity_column} < {-polarity_threshold}"]):
        indices = comment_df.query(query).index.values
        terms[polarity_name] = tfidf_df.iloc[indices] \
            .mean(axis=0) \
            .sort_values(ascending=False)
            
    return terms
