"""Translate via llm."""

import asyncio
import os

import aiohttp
from rich.console import Console

HEADERS = {"Content-Type": "application/json", "accept": "application/json"}
HEADERS = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'any')}"}
JDATA = {}
console = Console()

TIMEOUT = aiohttp.ClientTimeout(total=120)

PROMPT = """\
You are a professional translator. Translate the following text to simplified Chinese. You must translate everything to simplified Chinese. Just provide the translated text. Do not provide anything else including any thinking process nor reasoning process nor other notes:
"""


async def aiohttp_get(url: str, headers: None | dict = None) -> str | Exception:
    """Get with aiohttp."""
    if headers is None:
        headers = HEADERS
    async with aiohttp.ClientSession() as sess:
        res = await sess.get(url, headers=headers)
        res.raise_for_status()
        return await res.text()


async def aiohttp_post(
    url: str,
    jdata: None | dict = None,
    headers: None | dict = None,
    timeout: float | aiohttp.ClientTimeout = TIMEOUT,
) -> str | Exception:
    """Get with aiohttp."""
    if jdata is None:
        jdata = JDATA
    if headers is None:
        headers = HEADERS

    if isinstance(timeout, float):
        timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        res = await sess.post(url, json=jdata, headers=headers)
        res.raise_for_status()
        return await res.json()


async def llm_tr(  # pylint: disable=too-many-arguments
    text: str,
    llm_model: str = "gpt-4o-mini",
    prompt: str = PROMPT,
    timeout: float | aiohttp.ClientTimeout = TIMEOUT,
    base_url: None | str = None,
    api_key: None | str = None,
) -> str | Exception:
    """Translate via llm."""
    if not base_url:
        base_url = f'{os.getenv("OPENAI_BASE_URL", "").strip("/")}'

    base_url = base_url.strip("/")
    if not base_url.endswith("v1"):
        base_url = f"{base_url}/v1"

    url = f"{base_url}/chat/completions"

    if api_key:
        headers = {"Authorization": f"Bearer {api_key}"}
    else:
        headers = HEADERS

    jdata = {
        "messages": [{"role": "user", "content": f"{prompt!r} {text!r}"}],
        "model": llm_model,
    }
    res = await aiohttp_post(
        url,
        jdata=jdata,
        headers=headers,
        timeout=timeout,
    )
    llm_tr.json = res  # type: ignore
    return res["choices"][0]["message"]["content"].strip()  # type: ignore


if __name__ == "__main__":
    trtext = asyncio.run(llm_tr("This is a test."))

    console.print(trtext)
