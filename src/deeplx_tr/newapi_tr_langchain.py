"""
Translate via newapi api using langchain instead of direct openai.

Copy of newapi_tr0.py
also azure openai

"""
# pylint: disable=too-many-statements, too-many-branches, too-many-arguments

import asyncio
import os

from dotenv import load_dotenv
from httpx import Timeout
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from loguru import logger
from pydantic import BaseModel  # , SecretStr
# from pydantic.v1.types import SecretStr
from pydantic.types import SecretStr
from typing import Union
from ycecream import y

# from deeplx_tr.newapi_models import newapi_models

y.configure(sln=1, rn=1, st=1)  # e=0 to turn off output from y
# show_line_number/return_none=True, ennale=False, turn y off
# y.configure(sln=1, rn=1, e=0)


class SecretModel(BaseModel):
    secret: Union[SecretStr, str]


API_VERSION = "2024-03-01-preview"
load_dotenv()
BASE_URL = os.getenv("BASE_URL", "")
API_KEY = SecretModel(secret=os.getenv("API_KEY", ""))

async def newapi_tr_langchain(
    text: str,
    model: str = "gpt-3.5-turbo",
    base_url: str = "",
    api_key: Union[SecretStr, str] = "",
    to_lang: str = "Chinese",
    timeout=Timeout(45.0, connect=10.0),
    **kw,
) -> str:
    """
    Translate to to_lang.

    model: str = "gpt-3.5-turbo"
    base_url: str = ""
    api_key: str = ""
    to_lang: str = "Chinese"
    # timeout=Timeout(45.0, connect=10.0)

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
    # get in the same format as API_KEY
    try:
        api_key = SecretModel(secret=api_key)
    except Exception:
        api_key = SecretModel(secret="")

    if not base_url:
        base_url = "https://newapi.dattw.eu.org/v1"
    if not api_key.secret.get_secret_value():  # empty original api_key
        # load_dotenv()  # load .env, override=False, env var has precedence
        # api_key = os.getenv("API_KEY", "")
        api_key = API_KEY

    if not api_key:
        raise Exception(  # pylint: disable=broad-exception-raised
            "API_KEY not set. Set API_KEY in env var or in .env and try again."
        )

    subs = to_lang
    if to_lang.lower() in ["chinese"]:
        subs = f"simplified {to_lang}"

    content = f"""\
You are an expert {to_lang} translator. Your task is to translate \
TEXT into {to_lang}. You translate text into smooth and natural \
{subs} while maintaining the meaning in the original text. \
You only provide translated text. You do nor provide any explanation.
"""

    # TEXT: {text}

    if "azure.com" in base_url.lower():
        client = AzureChatOpenAI(
            timeout=timeout,
            api_version=API_VERSION,
            azure_endpoint=base_url,
            api_key=api_key.secret.get_secret_value(),  # type: ignore
        )
    else:
        client = ChatOpenAI(
            model=model,  # or deployment_name
            timeout=timeout,
            base_url=base_url,
            api_key=api_key.secret.get_secret_value(),
        )

    completion: BaseMessage = AIMessage(content="")
    # _ = """  # this hangs pytest
    completion = await client.ainvoke(
        [SystemMessage(content=content), HumanMessage(content=f"{text}\n{to_lang}: ")],
        **kw,
    )
    # """

    # print(completion.to_json())
    # print(completion)

    logger.trace(f"{completion=}")

    try:
        # trtext = completion.choices[0].message.content
        trtext = completion.content
    except Exception as exc:
        logger.trace(exc)
        raise

    return str(trtext)


async def main():
    """Bootstrap."""
    text = "Test this and that."
    y(text)

    api_key = "YourAPIKey"
    base_url = "https://mikeee-duck2api.hf.space/hf/v1"

    trtext = await newapi_tr_langchain(text, base_url=base_url, api_key=api_key)
    y(trtext)


if __name__ == "__main__":
    asyncio.run(main())
