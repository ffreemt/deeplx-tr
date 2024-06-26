"""
Translate via newapi api.

also azure openai

"""
# pylint: disable=too-many-statements, too-many-branches

import asyncio
import datetime
import os
from collections import defaultdict, deque
from contextlib import suppress
from pathlib import Path
from random import randrange
from time import monotonic
from typing import List

import diskcache
from httpx import Timeout
from loadtext import loadtext
from loguru import logger
from dotenv import load_dotenv

# from openai import OpenAI, AzureOpenAI
from openai import AsyncAzureOpenAI, AsyncOpenAI
from ycecream import y

from deeplx_tr.newapi_models import newapi_models

y.configure(sln=1)

API_VERSION = "2024-03-01-preview"

cache = diskcache.Cache(Path.home() / ".diskcache" / "newapi-tr")
# cache.set("models", [...])  # somewhere sometime

try:
    _ = list(cache.get("models"))  # type: ignore
    if not isinstance(_, (list, tuple)):
        _ = []
except:  # noqa
    _ = []
# DEQ = deque(_)
# DEQ = deque(newapi_models)
# DEQ = deque(newapi_models[:10])

# models = newapi_models[-10:]
models = newapi_models[:25]
DEQ = deque(models)

y(DEQ)

# initialize stats for each model
model_use_stats = {model: defaultdict(int) for model in DEQ}

async def cache_incr(item, idx, inc=1):
    """Increase cache.get(item)[idx] += inc."""
    _ = cache.get(item)
    # silently ignore all exceptions
    try:
        # _[idx] = _[idx] + inc
        _[idx] += inc  # type: ignore
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(exc)
        return

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, cache.set, item, _)


async def newapi_tr(
    text: str,
    model: str = "gpt-3.5-turbo",
    base_url: str = "",
    api_key: str = "",
    to_lang: str = "Chinese",
    timeout=Timeout(45.0, connect=10.0),
    **kw,
) -> str:
    """
    Translate to to_lang.

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
    if not base_url:
        base_url = "http://newapi.dattw.eu.org/v1"
    if not api_key:
        load_dotenv()  # load .env, override=False, env var has precedence
        api_key = os.getenv("API_KEY", "")

    if not api_key:
        raise Exception(
            "API_KEY not set. " "Set API_KEY in env var or in .env and try again."
        )

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
    # print(completion.to_json())
    # print(completion)

    logger.trace(f"{completion=}")

    try:
        trtext = completion.choices[0].message.content
    except Exception as exc:
        logger.trace(exc)
        raise

    return trtext


async def worker(
    queue_texts,
    deq_models,
    wid: int = -1,
    model_suffix: bool = True,
    queue_trtexts=asyncio.Queue(),
    timeout: float = -1,
) -> List[str]:
    """
    Consume/translate texts queue_texts to generate queue_trtexts and trtext_list.

    Args:
    ----
    queue_texts: async queue for texts
    deq_models: deuque for models
    wid: identifier of this worker for debugging and info collecting, default randrange(1000)
    queue_trtexts: async queue for translated texts, shared among possible many workers
    timeout: seconds per item in queue_texts, to exit While True loop to prevent hanging, default 30

    """
    if wid < 0:
        wid = randrange(1000)
    if timeout < 0:
        timeout = 30
    n_items = queue_texts.qsize()
    then = monotonic()
    trtext_list = []
    while True:
        if queue_trtexts.qsize() >= n_items or monotonic() - then > timeout * n_items:
            break
        try:
            seqno_text = queue_texts.get_nowait()
            logger.trace(f"{seqno_text=}")

            # there is no need for this 'if', but just to play safe
            if len(seqno_text) == 2:
                seqno, text = seqno_text
            else:
                text = str(seqno_text)
                seqno = -1
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.1)
            continue  # other worker may fail and put back items to the queue_texts
        except Exception as exc:
            logger.warning(f"{exc=}")
            raise

        # process text, output from queue_texts.get_nowait()
        model = deq_models[-1]
        deq_models.rotate()
        logger.trace(f" deq_models rotated: {deq_models=}")
        logger.trace(f" {model=}")
        logger.trace(f" {text=}")

        # translate using async newapi_tr
        try:
            logger.trace(f" try newapi_tr {text=} {wid=}")
            trtext = await newapi_tr(text, model=model)
            logger.trace(f" done newapi_tr {text=} {wid=}")
        except Exception as exc:  # pylint: disable=broad-except
            logger.trace(f"{exc=}, {wid=}")
            trtext = exc  # for retry in the subsequent round

            # optinally, remove model from deq_models since it does not deliver
            try:
                # deq_models.remove(model)
                ...
            except:  # pylint: disable=broad-except  # noqa
                ...
                # maybe another worker already did deq_models.remove(model)
                ...
        except:  # to make pyright happy  # noqa
            ...
            trtext = Exception("bare except exit")
        finally:
            logger.trace(f" {trtext=}  {wid=} ")
            queue_texts.task_done()
            logger.trace(f"\n\t >>====== que.task_done()  {wid=}")

            # put text back in the queue_texts if Exception
            if isinstance(trtext, Exception):
                logger.trace(
                    f"{wid=} {seqno=} failed {trtext=}, back to the queue_texts"
                )
                await queue_texts.put((seqno, text))
                await cache_incr("workers_fail", wid)

                model_use_stats[model]["fail"] += 1

                await asyncio.sleep(0.1)  # give other workers a chance to try
            else:
                # text not empty but text.strip() empty, try gain
                if text.strip() and not trtext.strip():
                    logger.trace(
                        f"{wid=} {seqno=} empty trtext, back to the queue_texts"
                    )
                    # try again if trtext empty
                    await queue_texts.put((seqno, text))
                    await cache_incr("workers_emp", wid)

                    model_use_stats[model]["empty"] += 1

                    await asyncio.sleep(0.1)
                else:
                    logger.info(f"{wid=} {seqno=} done ")

                    if model_suffix:
                        trtext = f"{trtext} [{model}]"

                    trtext_list.append((seqno, trtext))
                    await queue_trtexts.put((seqno, trtext))
                    await cache_incr("workers_succ", wid)

                    model_use_stats[model]["succ"] += 1

    logger.trace(f"\n\t {trtext_list=}, {wid=} fini")

    return trtext_list


async def batch_newapi_tr(texts: List[str], n_workers: int = 4, model_suffix: bool = True):
    """
    Translate in batch using urls from deq.

    Args:
    ----
        texts: list of text to translate
        n_workers: number of workers

    Returns:
    -------
        list of translated texts

    refer to python's official doc's example and asyncio-Queue-consumer.txt.

    """
    try:
        n_workers = int(n_workers)
    except Exception:  # pylint: disable=broad-except
        n_workers = 4
    if n_workers == 0:
        n_workers = len(texts)
    elif n_workers < 0:
        n_workers = len(texts) // 2

    # cap to len(texts)
    n_workers = min(n_workers, len(texts))

    logger.info(f"{n_workers=}")

    # logger.debug(y(n_workers))

    que = asyncio.Queue()
    for idx, text in enumerate(texts):
        await que.put((idx, text))  # attach seq no for retry

    logger.trace(f"{que=}")

    cache.set("workers_succ", [0] * n_workers)
    cache.set("workers_fail", [0] * n_workers)
    cache.set("workers_emp", [0] * n_workers)

    tasks = [asyncio.create_task(worker(que, DEQ, _, model_suffix=model_suffix)) for _ in range(n_workers)]

    logger.trace("\n\t >>>>>>>> Start await asyncio.gather")
    trtext_list = await asyncio.gather(*tasks)
    logger.trace("\n\t  >>>>>>>> Done await asyncio.gather")

    logger.trace(f"{trtext_list=}")

    trtext_list1 = []
    for _ in trtext_list:
        trtext_list1.extend(_)  # type: ignore
    logger.trace(f"{trtext_list1=}")

    succ = cache.get("workers_succ")
    logger.info(f"""success\n\t {succ}, {sum(succ)}""")  # type: ignore
    logger.info(f"""failure\n\t {cache.get("workers_fail")}""")
    logger.info(f"""empty\n\t {cache.get("workers_emp")}""")

    return trtext_list1


async def main():
    """Test and check models."""
    text = "A digital twin is a virtual model or representation of an object,\
    component, or system that can be updated through real-time data via \
    sensors, either within the object itself or incorporated into the \
    manufacturing process."

    then = monotonic()
    # coros = [newapi_tr(text), newapi_tr(text, temperature=0.35)] * 4

    # https://linux.do/t/topic/97173
    models = [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-uuci",
        "gpt-4-turbo-preview-uuci",
        "gpt-4-uuci",
        "gpt-4-turbo-2024-04-09-furry",
        "gpt-4o-furry",
        "gpt-3.5-turbo-furry",
        "gpt-4-0o0",
        "gpt-4o-0o0",
        "gpt-4-turbo-0o0",
        "claude-3-opus-20240229-0o0",
        # "claude-3-sonnet-20240229-0o0",
        # "claude-3-haiku-20240307-0o0",
        # "claude-2.0-0o0",
        # "zephyr-0o0",
        "kimi-0o0",
        "kimi-vision-0o0",
        # "reka-core-0o0",
        # "reka-flash-0o0",
        # "reka-edge-0o0",
        "command-r-0o0",
        "command-r-plus-0o0",
        "deepseek-chat-0o0",
        # "deepseek-coder-0o0",
        "google-gemini-pro-0o0",
        # "gemma-7b-it-0o0",
        # "llama2-7b-2048-0o0",
        "llama2-70b-4096-0o0",
        # "llama3-8b-8192-0o0",
        # "llama3-70b-8192-0o0",
        "mixtral-8x7b-32768-0o0",
        # "gpt-3.5-turbo-hf",
        # "gpt-3.5-turbo-edwagbb",
        # "gpt-3.5-turbo-smgc",
        # "gpt-3.5-turbo-sfun",
        # "gpt-3.5-turbo-neuroama",
        "gpt-3.5-turbo-pzero",
        "gpt-4o-pzero",
        "gpt-4-turbo-pzero",
        "deepseek-chat",  # 10cny 2024-06-11
    ]

    # models = ["deepseek-chat", "deepseek-chat-0o0"]
    # models = ["gpt-4-turbo-pzero"]
    # models = ["gpt-4o-pzero", "gpt-3.5-turbo-pzero"]
    # models = ['gpt-4-turbo-2024-04-09-furry', 'gpt-4o-furry', 'gpt-3.5-turbo-furry']

    coros = [newapi_tr(text, model=model) for model in models]

    trtexts = await asyncio.gather(*coros, return_exceptions=True)

    for idx, (model, trtext) in enumerate(zip(models, trtexts), 1):
        print("\n", idx, model)
        print("\t", trtext)

    print(f"{monotonic() - then :.2f}s")


if __name__ == "__main__":
    # asyncio.run(check())
    # asyncio.run(main())

    # texts = loadtext(r"C:\syncthing\00xfer\2021it\2024-05-30.txt", splitlines=1)
    texts = loadtext(r"C:\syncthing\00xfer\2021it\2024-06-20.txt", splitlines=1)

    today = f"{datetime.date.today()}"
    with suppress(Exception):
        texts = loadtext(rf"C:\syncthing\00xfer\2021it\{today}.txt", splitlines=1)

    texts = texts[:30]

    # _ = asyncio.run(batch_newapi_tr(["test 123", "test abc "]))

    then = monotonic()

    # trtexts = asyncio.run(batch_newapi_tr(texts))

    n_workers = len(DEQ)
    y(n_workers)

    trtexts = asyncio.run(batch_newapi_tr(texts), )

    trtexts = sorted(trtexts, key=lambda x: x[0])

    # print(''.jount(trtexts))
    print('\n\n'.join([', '.join(map(str, elm)) for elm in trtexts]))

    print(f"{monotonic() - then :.2f}s")

    stats =  [[*map(lambda x: model_use_stats[model][x], ['succ', 'fail', 'empty'])] for model in DEQ]
    # y(stats)
    print(f"{'model':<28}:", ["success", "failure", "empty"])
    for model, _ in zip(models, stats):
        print(f"{model:<28}:", _)
