"""Web search tool backed by the Google Custom Search JSON API."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import requests

TOOL_DEFINITION: Dict[str, Any] = {
    "name": "web_search",
    "description": "Search the web for current information on a topic",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string",
            },
            "max_results": {
                "type": "integer",
                "default": 5,
                "description": "Maximum number of results",
            },
        },
        "required": ["query"],
    },
}

# Required environment variables:
#   GOOGLE_API_KEY          – API key from Google Cloud Console
#   GOOGLE_CSE_ID           – Custom Search Engine ID (cx)
_API_KEY_VAR = "GOOGLE_API_KEY"
_CSE_ID_VAR = "GOOGLE_CSE_ID"
_ENDPOINT = "https://www.googleapis.com/customsearch/v1"


def execute(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Hit the Google Custom Search API and return structured results."""
    api_key = os.environ.get(_API_KEY_VAR)
    cse_id = os.environ.get(_CSE_ID_VAR)

    if not api_key or not cse_id:
        return [
            {
                "title": "Search unavailable",
                "url": "",
                "snippet": (
                    f"Set {_API_KEY_VAR} and {_CSE_ID_VAR} environment variables. "
                    f"See https://developers.google.com/custom-search/v1/overview"
                ),
            }
        ]

    try:
        resp = requests.get(
            _ENDPOINT,
            params={
                "key": api_key,
                "cx": cse_id,
                "q": query,
                "num": min(max_results, 10),  # API caps at 10 per request
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        results: List[Dict[str, str]] = []
        for item in data.get("items", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                }
            )

        return results[:max_results]

    except Exception as e:
        return [{"title": "Search unavailable", "url": "", "snippet": f"Error: {e}"}]
