"""
Translate via deepl, based on deeplx, credits to 小芍同学 and chatgpt.

Supported languages: 'de', 'en', 'es', 'fr', 'it', 'ja', 'ko', 'nl',
  'pl', 'pt', 'ru', 'zh','bg', 'cs', 'da', 'el', 'et', 'fi', 'hu', 'lt',
  'lv', 'ro', 'sk', 'sl', 'sv'
"""

# pylint: disable=invalid-name, line-too-long, broad-exception-raised
# pip install set-loglevel loguru icecream httpx pyperclip typer
# pip install nuitka ordered-set
import json
import os
import random

# import string
import sys
import time
from typing import List, Optional

# import gjson
# import requests
import httpx
import pyperclip
import typer

# import pygments  # for nuikta packaging
# from icecream import ic
from loguru import logger
from set_loglevel import set_loglevel

url = "https://www2.deepl.com/jsonrpc"

if os.environ.get("LOGURU_LEVEL") is None:
    logger.remove()
    logger.add(sys.stderr, level=set_loglevel())

_ = """
ic.configureOutput(
    includeContext=True,
    outputFunction=logger.debug,  # outputFunction=logger.info,
)
# """
__version__ = "0.0.1"
lang_list = [
    "de",
    "en",
    "es",
    "fr",
    "it",
    "ja",
    "ko",
    "nl",
    "pl",
    "pt",
    "ru",
    "zh",
    "bg",
    "cs",
    "da",
    "el",
    "et",
    "fi",
    "hu",
    "lt",
    "lv",
    "ro",
    "sk",
    "sl",
    "sv",
]

app = typer.Typer(
    name="deeplx-tr",
    add_completion=False,
    help=f"deeplx translate, supported languages: {lang_list}",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{app.info.name} v.{__version__} -- {__doc__}")
        raise typer.Exit()


def init_data(source_lang: Optional[str], target_lang: Optional[str]):
    """Prep initial data."""
    return {
        "jsonrpc": "2.0",
        "method": "LMT_handle_texts",
        "id": random.randint(100000, 109999) * 1000,
        "params": {
            "splitting": "newlines",
            "lang": {
                "source_lang_user_selected": source_lang,
                "target_lang": target_lang,
            },
        },
    }


def get_i_count(translate_text: str) -> int:
    """Count i."""
    return translate_text.count("i")


def get_random_number() -> int:
    """Gen a random number."""
    return random.randint(100000, 109999) * 1000


def get_timestamp(i_count: int) -> int:
    """Gen timestamp."""
    ts = int(time.time() * 1000)
    if i_count != 0:
        i_count = i_count + 1
        return ts - ts % i_count + i_count

    # else:
    return ts


# def deeplx_tr(data: dict) -> dict:
def deeplx_tr(
    text: str,
    source_lang: str = "auto",
    target_lang: str = "zh",
) -> dict:
    """Translate."""
    # source_lang = data.get("source_lang")
    # target_lang = data.get("target_lang", "zh")

    if source_lang is not None:
        source_lang = source_lang.lower()
    if target_lang is not None:
        target_lang = target_lang.lower()

    # translate_text = data.get("text", "")
    translate_text = text[:]

    # handle whitespace input and non-string input
    try:
        translate_text = translate_text.strip()
    except Exception:
        translate_text = str(translate_text)

    if not translate_text:
        return {"data": ""}

    if source_lang not in lang_list:
        source_lang = "auto"
    if target_lang not in lang_list:
        if source_lang not in ["zh"]:
            target_lang = "zh"
        else:
            target_lang = "en"

    post_data = init_data(source_lang, target_lang)
    text_ = {
        "text": translate_text,
        # "requestAlternatives": 3,
    }
    post_data["params"]["texts"] = [text_]
    post_data["params"]["timestamp"] = get_timestamp(get_i_count(translate_text))
    post_str = json.dumps(post_data)
    if (post_data["id"] + 5) % 29 == 0 or (post_data["id"] + 3) % 13 == 0:
        post_str = post_str.replace('"method":"', '"method" : "', 1)
    else:
        post_str = post_str.replace('"method":"', '"method": "', 1)

    logger.debug(post_str)

    try:
        # response = requests.post(url, post_str, headers={'Content-Type': 'application/json'})
        response = httpx.post(
            url,
            data=post_str,
            headers={"Content-Type": "application/json"},  # type: ignore
        )

        # response = requests.post(url, post_data, headers={'Content-Type': 'application/json'})
        # print(response.text)
        # response_json = json.loads(response.text)
        response_json = response.json()
    except Exception as e:
        raise e

    if "error" in response_json:
        error = response_json["error"]
        raise Exception(error.get("message"))

    # else:
    return {
        "code": 200,
        "id": post_data["id"],
        "data": response_json["result"]["texts"][0]["text"],
    }


@app.command()
def main(
    version: Optional[bool] = typer.Option(  # pylint: disable=(unused-argument
        None,
        "--version",
        "-v",
        "-V",
        help="Show version info and a list of supported languages and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
    text: Optional[List[str]] = typer.Argument(
        None,
        help="Source text.",
    ),
    clipb: Optional[bool] = typer.Option(
        None,
        "--clipb",
        "-c",
        help="Use clipboard content if set or if `text` is empty.",
    ),
    from_lang: Optional[str] = typer.Option(
        None, "--from-lang", "-f", help="Source language."
    ),
    to_lang: Optional[str] = typer.Option(
        "zh", "--to-lang", "-t", help="Target language."
    ),
):
    """
    deeplx in plain python.

    supported languages: {lang_list}

    set LOGURU_LEVEL=DEBUG or set LOGLEVEL=10 to turn on debug|verbose mode.
    """
    logger.debug(["text", "clipb", "from_lang", "to_lang"])
    logger.debug(text)
    logger.debug(clipb)
    logger.debug(from_lang)
    logger.debug(to_lang)

    # ic(text)

    # collect text
    # check clipb first, then terminal
    # if nothing supplied from terminal, check clipb again
    if clipb:
        text_str = pyperclip.paste()
    else:
        if text is None:
            text_str = ""
        else:
            text_str = " ".join(text).strip()
        if not text_str:
            text_str = pyperclip.paste()

    logger.debug(text_str)
    # ic(text_str)

    if not text_str:
        _ = "Either supply some text from the terminal or set -c (--clipb) to translate clipboard content (make sure the clipboard is not empty)."
        msg = typer.style(_, fg=typer.colors.GREEN, bold=True)
        typer.echo(msg)
        _ = input("Press Enter to continue...")
        raise typer.Exit(code=1)

    data = {
        "text": text_str,
        "source_lang": from_lang,
        "target_lang": to_lang,
    }

    try:
        resp = deeplx_tr(**data)
    except Exception as e:
        logger.error(e)
        raise typer.Exit(code=1)

    logger.debug(resp)
    # ic(resp)

    _ = resp.get("data", "").strip()
    typer.echo(_)

    if _:
        pyperclip.copy(_)


if __name__ == "__main__":
    # uvicorn.run(app="main:app", host="127.0.0.1", port=8080, reload=True)
    # print(sys.argv[1:], bool(sys.argv[1:]), _)
    _ = """
    if sys.argv[1:]:
        _ = {"text": " ".join(sys.argv[1:])}
    else:
        _ = {"text": "this is a test"}

    try:
        res = deeplx_tr(_)
        print(_, res)
    except Exception as exc:
        print('exc: ', exc)
        logger.error(exc)
    # """
    app()
