"""src.api.api_methods.py -- methods for startup of the FastAPI"""

import sys
sys.path.insert(0, "../.")
sys.path.insert(0, "src")

import logging

from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.comment_analysis.comment_processor import CommentProcessor
from src.comment_analysis.exceptions import NoCommentsFoundException
from api.api_schema import CommentAnalysisRequest, AnalysisResponse, CommentExtractionRequest, CommentExtractionResponse


# logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# API setup

app = FastAPI()  


@app.get('/health',
         description='Check the health of the API.')
async def get_response():
    return "Comment API is healthy"


@app.post("/extract-comments",
          description='Extract comments from raw HTML data.')
async def extract_comments(extraction_request: CommentExtractionRequest) -> JSONResponse:
    """
    Endpoint for the extraction of comments from raw HTML data.
    
    :param extraction_request: extraction request parameters in CommentExtractionRequest format.
    
    :return: list of extracted comments in a JSONResponse object.
    """
    try:
        comments = CommentProcessor.get_comments_from_html(
            soup=BeautifulSoup(extraction_request.raw_html, 'html.parser')
        )
    except NoCommentsFoundException:
        logger.error(f"NoCommentsFoundException: {e}")
        return JSONResponse(status_code=422,
                            content="No/insufficient raw HTML provided.")
        
    logger.info(f"Successfully extracted {len(comments)} comments from raw HTML.")
    return CommentExtractionResponse(comments=comments)

@app.post("/analyze-comments",
          description='Perform a complete analysis of comments, donations, and likes/dislikes based on raw HTML data.')
async def analyze_comments(analysis_request: CommentAnalysisRequest) -> JSONResponse:
    """
    Endpoint for the complete analysis of uploaded raw HTML data.

    :param analysis_request: JSON data of request in CommentAnalysisRequest format.

    :return: returns a JSONResponse with metrics on the analyzed data or with an error description.
    """
    request_params = analysis_request.model_dump()
    
    # check if the text is empty
    if not request_params['raw_html'].strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
    
    # run analysis
    processor = CommentProcessor(**request_params['params'])

    try:
        response_dict = processor.process_data(raw_html=request_params['raw_html'])
    except NoCommentsFoundException as e:
        logger.error(f"NoCommentsFoundException: {e}")
        return JSONResponse(status_code=422,
                            content="No/insufficient raw HTML provided.")

    response = {}
    for key in ['dislikes', 'donations', 'comments']:
        if key in response_dict.keys():
            response.update({key: response_dict[key]['metrics']})

    return AnalysisResponse(**response)
