"""src.api.api_methods.py -- methods for startup of the FastAPI"""

import sys
sys.path.insert(0, "../.")
sys.path.insert(0, "src")

import logging

from fastapi import FastAPI, HTTPException

from comment_analysis.comment_processor import CommentProcessor
from api.api_schema import AnalysisRequest, AnalysisResponse


# logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# API setup

app = FastAPI()  


@app.get('/health')
async def get_response():
    return "Comment API is healthy"


@app.post("/analyze-comments",
          response_model=AnalysisResponse)
async def analyze_comments(analysis_request: AnalysisRequest):
    """
    Endpoint for the analysis of uploaded comments.

    :param raw_html: HTML data to analyse (HTML body of YouTube video
    :param analysis_params: optional dictionary with parameters for the analysis via CommentProcessor
    """
    request_params = analysis_request.dict()
    
    # check if the text is empty
    if not request_params['raw_html'].strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    logger.info(f"{type(request_params)=}, {type(request_params['params'])=}, {request_params['params']=}")
    
    # run analysis
    processor = CommentProcessor(**request_params['params'])

    response_dict = processor.process_comments(raw_html=request_params['raw_html'])

    response = {}
    for key in ['dislikes', 'donations', 'comments']:
        if key in response_dict.keys():
            response.update({key: response_dict[key]['metrics']})

    return response
