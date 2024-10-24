"""src.comment_analysis.visualisation.py -- methods for analysis output visualiation."""

import logging
import pandas as pd
import random
import seaborn as sns

from matplotlib import pyplot as plt
from wordcloud import WordCloud


logger = logging.getLogger(__name__)


# WordCloud generation

def custom_color_func(*args, 
                      **kwargs) -> str:
    """
    Controls the color of the WordCloud based on a custom color function, following the HSL format (hue, saturation, brightness).

    :return: HSL color code string.
    """
    # HSL format to control color tone
    hue = 200  # constant hue
    saturation = random.randint(40, 50)  # lower saturation for less vibrant colors (limits: 0 - 100)
    lightness = random.randint(20, 80)   # color brightness (limits: 0 - 100, lower is darker)
    
    return f"hsl({hue}, {saturation}%, {lightness}%)"


def generate_wordcloud(tfidf_scores: pd.Series,
                       title: str,
                       tfidf_score_threshold: float = 0.1,
                       hide_output: bool = False) -> None:
    """
    Generate a WordCloud based on a series of TF-IDF scores.

    :param tfidf_scores: pd.Series with scores.
    :param title: title for the word cloud.
    :param tfidf_score_threshold: threshold for scores to filter on.
    :param hide_output: boolean to trigger while testing to hide the ouptut.
    """
    sns.set_theme(font_scale=1.3)
    
    tfidf_scores_filtered = tfidf_scores[tfidf_scores > tfidf_score_threshold]
    
    if len(tfidf_scores_filtered) == 0:
        logger.warning(f"No residual terms for TF-IDF score threshold of {tfidf_score_threshold} and WordCloud for {title}.")

    wordcloud = WordCloud(width=600, 
                          height=400, 
                          background_color='white',
                          random_state=42,
                          color_func=custom_color_func).generate_from_frequencies(tfidf_scores)
    
    plt.figure(figsize=(6, 3))
    plt.title(title)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    if not hide_output:
        plt.show()


# Plotting of results

def plot_results(response_dict: dict,
                 plot_word_cloud: bool = True,
                 word_cloud_terms: int = 10,
                 tfidf_score_threshold: float = 0.1,
                 hide_output: bool = False) -> None:
    """
    Plot results of comment analysis.

    :param response_dict: dictionary with CommentProcessor response.
    :param plot_word_cloud: show WordCloud of relevant terms.
    :param word_cloud_terms: number of terms to plot in each word cloud.
    :param tfidf_score_threshold: threshold for TF-IDF scores to filter for.
    :param hide_output: boolean to trigger while testing.
    """
    sns.set_theme(font_scale=1.3)
    
    if 'donations' in response_dict.keys():
        response_dict['donations']['details']['donation_df']['conv_donation'].hist(bins=30)
        target_currency = response_dict['donations']['metrics']['donations__currency']
        donations_total = round(response_dict['donations']['details']['donation_df']['conv_donation'].sum(), 2)
        
        plt.xlabel(f'Donation [{target_currency}]')
        plt.ylabel('Count [-]')
        plt.title(f'Donations\nTotal: {donations_total} {target_currency}')
        if not hide_output:
            plt.show()

    if 'comments' in response_dict.keys():
        # plot overall comment polarity category bar chart
        # melt the DF so we can plot both columns as categorical variables in the same plot
        df_melted = response_dict['comments']['details']['comment_df'].melt(value_vars=['sentiment_textblob', 'sentiment_vader'], 
                                    var_name='Analysis Method', 
                                    value_name='sentiment')

        # plot using matplotlib / seaborn
        plt.figure(figsize=(8, 6))
        order = sorted(df_melted['sentiment'].unique())
        
        sns.countplot(x='sentiment', 
                      hue='Analysis Method',
                      order=order,
                      data=df_melted, 
                      dodge=True)

        plt.suptitle('Comment Polarity')
        plt.title('Categorical Distribution of Sentiment (TextBlob vs VADER)')
        plt.xlabel('Sentiment')
        plt.ylabel('Count [-]')
        if not hide_output:
            plt.show()
            
        # plot individual comment polarity histogram
        fig, ax = plt.subplots(figsize=(8, 6))
        response_dict['comments']['details']['comment_df']['polarity_textblob'].hist(
           bins=30,
           ax=ax, 
           label='textblob', 
           alpha=0.7
        )
        response_dict['comments']['details']['comment_df']['polarity_vader'].hist(
            bins=30, 
            ax=ax, 
            label='vader', 
            alpha=0.7
        )

        plt.suptitle('Comment Polarity')
        plt.title('Comment Polarity Histogram')
        plt.xlabel('Polarity [-]')
        plt.ylabel('Count [-]')
        plt.legend()
        if not hide_output:
            plt.show()
            
        # word clouds
        if plot_word_cloud:
            generate_wordcloud(
                response_dict['comments']['details']['relevant_terms']['positive'].head(word_cloud_terms),
                title='Positive Comments',
                tfidf_score_threshold=tfidf_score_threshold,
                hide_output=hide_output
            )
            generate_wordcloud(
                response_dict['comments']['details']['relevant_terms']['negative'].head(word_cloud_terms),
                title='Negative Comments',
                tfidf_score_threshold=tfidf_score_threshold,
                hide_output=hide_output
            )