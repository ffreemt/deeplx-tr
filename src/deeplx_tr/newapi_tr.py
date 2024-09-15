"""
Translate via newapi api.

also azure openai

"""
# pylint: disable=too-many-statements, too-many-branches, too-many-arguments

import os
from collections import deque
from pathlib import Path

import diskcache
from dotenv import load_dotenv
from httpx import Timeout
from loguru import logger

# from openai import OpenAI, AzureOpenAI
from openai import AsyncAzureOpenAI, AsyncOpenAI
from ycecream import y

from deeplx_tr.newapi_models import newapi_models

y.configure(sln=1, rn=1)

# show_line_number/return_none=True, ennale=False, turn y off
# y.configure(sln=1, rn=1, e=0)

API_VERSION = "2024-03-01-preview"

cache = diskcache.Cache(Path.home() / ".diskcache" / "newapi-tr")
# cache.set("models", [...])  # somewhere sometime

try:
    _ = list(cache.get("models"))  # type: ignore
    if not isinstance(_, (list, tuple)):
        _ = []
except:  # noqa  # pylint: disable=bare-except
    _ = []
# DEQ = deque(_)
# DEQ = deque(newapi_models)
# DEQ = deque(newapi_models[:10])

# models = newapi_models[-10:]
MODELS = newapi_models[:]
DEQ = deque(MODELS)

# y(DEQ)


async def newapi_tr(
    text: str,
    model: str = "gpt-3.5-turbo",
    base_url: str = "",
    api_key: str = "",
    to_lang: str = "Chinese",
    timeout=Timeout(10.0, connect=10.0),
    **kw,
) -> str:
    """
    Translate to to_lang.

    model: str = "gpt-3.5-turbo"
    base_url: str = ""
    api_key: str = ""
    to_lang: str = "Chinese"
    timeout=Timeout(45.0, connect=10.0)

    Args:
    ----
    text: string to translate
    base_url: base_url or endpoint for azure
    model: model name or deployment_name for azure
    api_key: api_key
    to_lang: destination language
    timeout: httpx.Timeout to control various timeouts
    **kw: additional params, e.g., temperature, repetition_penalty

    """
    load_dotenv()  # load .env, override=False, env var has precedence
    if not base_url:
        # base_url = "https://newapi.dattw.eu.org/v1"
        base_url = os.getenv("BASE_URL", "https://newapi.dattw.eu.org/v1")
    if not api_key:
        api_key = os.getenv("API_KEY", "NA")

    _ = """
    if not api_key:
        raise Exception(  # pylint: disable=broad-exception-raised
            "API_KEY not set. Set API_KEY in env var or in .env and try again."
        )
    # """

    if not base_url.rstrip("/").endswith("v1"):
        base_url = base_url + "/v1"

    subs = to_lang
    if to_lang.lower() in ["chinese"]:
        subs = f"simplified {to_lang}"

    content = f"""\
You are an expert {to_lang} translator. Your task is to translate \
TEXT into {to_lang}. You translate TEXT into smooth and natural \
{subs} while maintaining the meaning in the original text. \
You only provide translated text. You do nor provide any explanation.

TEXT: {text}"""

    if "azure.com" in base_url.lower():
        client = AsyncAzureOpenAI(
            timeout=timeout,
            api_version=API_VERSION,
            azure_endpoint=base_url,
            api_key=api_key,
        )
    else:
        client = AsyncOpenAI(
            timeout=timeout,
            base_url=base_url,
            api_key=api_key,
        )

    completion = ""
    # _ = """  # this hangs pytest
    completion = await client.chat.completions.create(
        model=model,  # or deployment_name
        messages=[
            {
                "role": "user",
                "content": content,
            },
        ],
        **kw,
    )
    # """

    # print(completion.to_json())
    # print(completion)

    logger.trace(f"{completion=}")

    try:
        trtext = completion.choices[0].message.content
    except Exception as exc:
        logger.trace(exc)
        raise

    return trtext
