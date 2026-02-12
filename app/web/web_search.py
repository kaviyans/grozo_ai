from tavily import TavilyClient
import os

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


# ---------- TAVILY WEB SEARCH ----------
async def web_search(query: str, max_results: int = 3):
    """
    Web search optimized for RAG fallback
    """
    response = tavily_client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
    )

    results = []

    if response.get("answer"):
        results.append(response["answer"])

    for r in response.get("results", []):
        results.append(
            f"{r['title']}\n{r['content']}"
        )

    return results


async def search_product_online(search_query: str):
    """
    Product discovery fallback using keyword query only
    """
    return web_search(search_query, max_results=5)

