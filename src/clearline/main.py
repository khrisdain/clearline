"""ASGI entry point: `uvicorn clearline.main:app`."""

from .api import create_app

app = create_app()
