# YouTube Comment Analysis

Repository for the analysis of YouTube comments based on raw HTML data. Extracts information on:

1. Donations on the video (converted to a target currency)
2. Translation of comments to English
3. Sentiment analysis on the comments
4. Extraction of relevant terms from extremely positive/negative comments
5. Estimation of the dislikes on a video based on likes and comment polarity

The repository provides a FastAPI to perform comment analysis as well as a class for this purpose (`CommentProcessor`).

## Preparation

Ensure that the body of the raw HTML of the YouTube page you want to inspect is pasted in `html_body.txt`. Scroll down a couple of times if you want to inspect more comments.

## Requirements

Ensure to install the requirements into a new virtual environment (`pip install -r requirements.txt`).

## Startup

You have multiple choices to execute the analysis:

1. FastAPI access: run the FastAPI and send analysis requests to it. This emulates application in a production environment. The API returns a JSON summary of the gathered metrics.
2. EDA / Method Development Notebook: investigate the individual steps of analysis by going through the `youtube-comment-analysis.ipynb` notebook (includes explanations and some consistency tests). Choose this path to understand the concept of the analysis and yield intermediates.
3. Productionized Analysis: Run the analysis through the CommentProcessor class via the `class-usage.ipynb` notebook (very compact notebook). This enables a condensed call to run the whole analysis. Choose this path to quickly gather all outputs of the analysis, including plots.
4. Command line access: Run `main.py` from the command line. You can adjust the source HTML path and plotting options via arguments. Choose for programmatic access independently from a Jupyter Notebook instance.

### FastAPI Access

The FastAPI app is stored in `src.api.app.py`. Start it using `uvicorn` via

```
uvicorn src.api.app:app --port 8000 --reload
```

The Swagger UI can then be accessed via:

```
localhost://127.0.0.1:8000/docs
```

Then, requests to the API can be submitted, e.g., via the requests library:

```python
import requests

response = requests.post(
    'localhost://127.0.0.1:8000/analyze-comments',
    json={
        "raw_html": "<enter raw HTML here>",
        "params": {
            "analyze_donations": True
        }
    }
)

if response.status_code == 200:
    print(response.content)
```

### Command Line Access

Submit queries via the command line, e.g., like this, specifying HTML data stored in a custom file:

```sh
python main.py --raw-html-path ./my-file.txt
```

List all available arguments by calling:

```sh
python main.py --help
```

### Tests

The repository is covered by a pytest test suite (functional / unit testing). This test suite can be executed by running `pytest tests  --cov src --cov-report html` from the root directory to yield test results and an HTML coverage report.
