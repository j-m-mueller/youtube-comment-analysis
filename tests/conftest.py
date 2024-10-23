"""tests.conftest.py -- global fixtures for tests."""

# path adjustments
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import uvicorn
import pytest
import time
import threading

from src.api.app import app


API_PORT = 8000
API_HOST = "127.0.0.1"


@pytest.fixture(scope="session", autouse=True)
def uvicorn_server():
    """Fixture to start a Uvicorn server using uvicorn.run in a separate thread."""
    def run_server():
        uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")
    
    # start the server in a new thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    time.sleep(5)
    
    # execute tests
    yield
