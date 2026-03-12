from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

def search_agent(state):

    query = state["search_results"][0]

    results = search.run(query)

    return {
        "search_results": [results]
    }