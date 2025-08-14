"""
API documentation module for ITS Camera AI.

Provides OpenAPI specifications, examples, and documentation generation.
"""

from .schemas import APIExamples, OpenAPIGenerator, api_examples

__all__ = [
    "OpenAPIGenerator",
    "APIExamples",
    "api_examples",
]
