"""Pytest configuration and fixtures."""

import numpy as np
import pytest


# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture
def mock_embedding():
    """Return a mock embedding that behaves like a numpy array."""
    embedding = np.array([0.1] * 384)
    return embedding
