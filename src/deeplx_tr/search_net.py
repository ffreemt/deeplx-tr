"""
Search query using tavily and searxng.

TAVILY_API_KEY
SEARX_HOST
"""
# pylint: disable=missing-function-docstring, broad-exception-raised

import os
from dotenv import load_dotenv
from langchain.utilities import SearxSearchWrapper
from tavily import TavilyClient

CLIENT = TavilyClient()
SEARCH = SearxSearchWrapper(k=5)
C_LIST = ["title", "url", "content", ]

load_dotenv()
assert os.getenv("TAVILY_API_KEY"), "Need to set env var TAVILY_API_KEY in .env or shell"
assert os.getenv("SEARX_HOST"), "Need to set env var SEARX_HOST in .env or shell"


def reorder(l_dict: list[dict], flag: bool = False) -> list[dict]:
    # delete raw_content if present
    for dict_ in l_dict:
        if "raw_content" in dict_:
            del dict_["raw_content"]

    reordered_list1 = [{key: d[key] for key in C_LIST} for d in l_dict]

    # only return the first part
    if flag:
        return reordered_list1

    reordered_list2 = [
        {key: d[key] for key in d.keys() if key not in C_LIST} for d in l_dict
    ]
    len_ = len(reordered_list1)
    for i in range(len_):
        reordered_list1[i].update(reordered_list2[i])

    # return reordered_list1

    # stringify all itmes
    return [{key: str(val) for key, val in d.items()} for d in reordered_list1]


# def tavily_search(query: str, max_results: int = 3) -> Optional[List[Dict]]:
def tavily_search(query: str, max_results: int = 3) -> None | list[dict]:
    # return CLIENT.search(query, max_results=max_results).get("results")
    res = CLIENT.search(query, max_results=max_results).get("results")

    if res is None:
        raise Exception("tavily returns None")

    return reorder(res)


def searx_search(query: str, max_results: int = 3, time_range="month"):
    res = SEARCH.results(query, num_results=max_results, time_range=time_range)

    # replace snippet with content, link with url in res
    for item in res:
        for elm1, elm2 in zip(["snippet", "link"], ["content", "url"]):
            item.update({elm2: item.get(elm1)})
            del item[elm1]

    return reorder(res)


def search_net(query: str, max_results: int = 3):
    try:
        query = str(query).strip()
    except Exception as e:
        query = ""

    if len(query) < 3:
        raise Exception("query too short (< 3), makes no sense to search")

    try:
        return tavily_search(query, max_results=max_results)
    except Exception:
        return searx_search(query, max_results=max_results)
