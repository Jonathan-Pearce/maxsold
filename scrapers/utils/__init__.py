"""Shared utilities for MaxSold scrapers"""

from .config import HEADERS, API_URLS
from .file_io import save_to_parquet, load_from_parquet, save_to_json, load_from_json

__all__ = [
    'HEADERS',
    'API_URLS',
    'save_to_parquet',
    'load_from_parquet',
    'save_to_json',
    'load_from_json',
]
