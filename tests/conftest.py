"""Shared test fixtures.

We set a fake OPENAI_API_KEY before importing anything from ``app`` so
Settings validation passes without a real key. Unit tests never make network
calls -- external services are replaced with mocks/fakes.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("ENVIRONMENT", "test")
