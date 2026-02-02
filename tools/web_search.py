import time
from typing import Dict, Any, List
from tools.base import BaseTool, ToolResult

class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the live web for factual information, current data, documentation, or entity facts. "
        "Returns top relevant snippet results with titles and URLs."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string."
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of search results to return (default 5)."
            }
        },
        "required": ["query"]
    }

    def __init__(self, default_max_results: int = 5):
        self.default_max_results = default_max_results

    def execute(self, query: str, max_results: int = None, **kwargs) -> ToolResult:
        start_time = time.perf_counter()
        if not query or not query.strip():
            return ToolResult(
                success=False,
                output="",
                error="Error: Empty search query provided.",
                execution_time=0.0
            )

        limit = max_results if max_results is not None else self.default_max_results
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query.strip(), max_results=limit))

            if not raw_results:
                return ToolResult(
                    success=True,
                    output="No search results found for query.",
                    execution_time=time.perf_counter() - start_time,
                    metadata={"count": 0}
                )

            formatted = []
            for idx, r in enumerate(raw_results, 1):
                title = r.get("title", "No Title")
                snippet = r.get("body", "")
                link = r.get("href", "")
                formatted.append(f"[{idx}] {title}\nURL: {link}\nSnippet: {snippet}")

            return ToolResult(
                success=True,
                output="\n\n".join(formatted),
                execution_time=time.perf_counter() - start_time,
                metadata={"count": len(raw_results)}
            )

        except Exception as e:
            # Fallback or informative error
            return ToolResult(
                success=False,
                output="",
                error=f"WebSearchError: Failed to fetch search results: {str(e)}",
                execution_time=time.perf_counter() - start_time
            )
