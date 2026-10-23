from typing import Any

from researchpilot.utils import CHARACTER_LIMIT, PAGE_CONTENT_LIMIT, get_tavily_client

__all__ = ["TavilySearchTools"]


class ResearchError(RuntimeError):
    """Raised when a topic cannot be researched."""

class TavilySearchTools:
    """A class to perform searches using the Tavily API."""

    def __init__(self):
        self.client = get_tavily_client()

    @staticmethod
    def _compact_results(
        results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Compact the search results to only include relevant information.

        Args:
            results (list[dict]): The search results.

        Returns:
            list[dict]: A list of compacted search results.
        """
        compacted = []

        for result in results:
            item = {
                key: result.get(key) 
                for key in (
                    "title",
                    "url",
                    "score",
                    "content",
                    "published_date",
                )
                if key in result
            }

            # Avoid sending excessively long content to the LLM.
            content = item.get("content")
            if isinstance(content, str):
                item["content"] = content[:CHARACTER_LIMIT]

            compacted.append(item)

        return compacted

    def web_search(self, query: str, max_results: int = 5) -> list[dict]:
        """Perform a web search using the Tavily API."""
        try:
            search_results = self.client.search(query, max_results=max_results)
            return self._compact_results(search_results.get("results", []))

        except (ConnectionError, TimeoutError) as e:
            raise ResearchError(f"Error during web search: {e}")

    def extract_page_content(self, urls: list[str]) -> dict[str, Any]:
        """Extract content from a web page using the Tavily API."""
        try:
            page_contents = self.client.extract(urls[:20])  # Limit to first 20 URLs (recommended by Tavily API)
            
            successful_results = []

            for page in page_contents.get("results", []):
                url = page.get("url")
                content = page.get("raw_content")

                if not url or not content:
                    continue  # Skip if URL or content is missing

                successful_results.append({
                    "url": url,
                    "raw_content": content[:PAGE_CONTENT_LIMIT],
                })

            return {
                "results": successful_results,
                "failed_results": page_contents.get("failed_results", []),
            }

        except (ConnectionError, TimeoutError) as e:
            raise ResearchError(f"Error during page content extraction: {e}")


    def map_website(self, url: str) -> list[dict]:
        """Map websites using the Tavily API."""
        try:
            mapped_websites = self.client.map(url)
            return mapped_websites.get("results", [])

        except (ConnectionError, TimeoutError) as e:
            raise ResearchError(f"Error during website mapping: {e}")

        except Exception as e:  # noqa: BLE001
            raise ResearchError(f"Unexpected error during website mapping: {e}")


    def crawl_website(self, url: str, max_depth: int = 2) -> list[dict]:
        """Crawl a website using the Tavily API."""
        try:
            crawled_websites = self.client.crawl(url, max_depth)
            results = []
            for page in crawled_websites.get("results", []):
                url = page.get("url")
                content = page.get("raw_content")

                if not url or not content:
                    continue  # Skip if URL or content is missing

                if isinstance(content, str):
                    content = content[:PAGE_CONTENT_LIMIT]

                results.append({
                    "url": url,
                    "raw_content": content,
                })

            return results

        except (ConnectionError, TimeoutError) as e:
            raise ResearchError(f"Error during website crawling: {e}")
