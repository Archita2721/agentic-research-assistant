import time

from langchain_community.tools import DuckDuckGoSearchRun

from constants import ENABLE_WEB_SEARCH

search = DuckDuckGoSearchRun()

def search_agent(state):

    if not ENABLE_WEB_SEARCH:
        return {"search_results": []}

    query = state["search_results"][0]

    start = time.perf_counter()
    results = search.run(query)
    print(f"[timing] web_search.run {(time.perf_counter() - start):.3f}s", flush=True)

    return {
        "search_results": [results]
    }
