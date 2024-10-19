# YouTube Comment Analysis

Repository for the analysis of YouTube comments. Extracts information on:

1. Donations on the video (converted to a target currency)
2. Translation of comments to English
3. Sentiment analysis on the comments
4. Extraction of relevant terms from extremely positive/negative comments
5. Estimation of the dislikes on a video based on likes and comment polarity

## Preparation

Ensure that the body of the raw HTML of the YouTube page you want to inspect is pasted in `html_body.txt`. Scroll down a couple of times if you want to inspect more comments.

## Requirements

Ensure to install the requirements into a new virtual environment (`pip install -r requirements.txt`).

## Startup

You have three choices to execute the analysis:

1. Investigate the individual steps by going through the `youtube-comment-analysis.ipynb` notebook (includes explanations and some consistency tests)
2. Run the analysis through the CommentProcessor class via the `class-usage.ipynb` notebook (very compact notebook)
3. Run `main.py` from the command line (no Jupyter environment required). You can adjust the source HTML path and plotting options via arguments.
