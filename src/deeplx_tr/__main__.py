"""Prep __main__.py."""

# pylint: disable=invalid-name
from typing import List, Optional

import pyperclip
import typer
from loguru import logger

# from set_loglevel import set_loglevel
from deeplx_tr import __version__, deeplx_tr, lang_list

app = typer.Typer(
    name="deeplx-tr",
    add_completion=False,
    help="deeplx-tr help",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(
            f"{app.info.name} v.{__version__} -- supported languages: {lang_list}"
        )
        raise typer.Exit()


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
    r"""
    deeplx translate in plain python.

    supported languages: {lang_list}

    set LOGURU_LEVEL=DEBUG or set LOGLEVEL=10 to turn on debug/verbose mode.
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
        _ = (
            "Either supply some text from the terminal or set -c (--clipb) "
            "to translate clipboard content (make sure the clipboard is not empty)."
        )
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
    app()
