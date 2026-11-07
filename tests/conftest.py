from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clearline.api import create_app
from clearline.config import Settings
from clearline.ingest import load_directory
from clearline.service import Clearline

SAMPLE = Path(__file__).parent.parent / "sample_data"


@pytest.fixture
def settings() -> Settings:
    return Settings(env="test", provider="fake", store="memory", auth_mode="dev", embedding_dim=512)


@pytest.fixture
def service(settings: Settings) -> Clearline:
    clearline = Clearline(settings)
    for document in load_directory(SAMPLE):
        clearline.ingest(document)
    return clearline


@pytest.fixture
def client(settings: Settings, service: Clearline) -> TestClient:
    return TestClient(create_app(settings, service))
