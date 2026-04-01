"""Web search tool backed by the DuckDuckGo Instant Answer API."""

from __future__ import annotations

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


def execute(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Hit the DuckDuckGo Instant Answer API and return structured results."""
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        results: List[Dict[str, str]] = []
        for topic in data.get("RelatedTopics", []):
            if "Text" in topic and "FirstURL" in topic:
                results.append(
                    {
                        "title": topic["Text"][:120],
                        "url": topic["FirstURL"],
                        "snippet": topic["Text"],
                    }
                )
            # Handle nested sub-topics
            elif "Topics" in topic:
                for sub in topic["Topics"]:
                    if "Text" in sub and "FirstURL" in sub:
                        results.append(
                            {
                                "title": sub["Text"][:120],
                                "url": sub["FirstURL"],
                                "snippet": sub["Text"],
                            }
                        )
            if len(results) >= max_results:
                break

        return results[:max_results]

    except Exception as e:
        return [{"title": "Search unavailable", "url": "", "snippet": f"Error: {e}"}]
