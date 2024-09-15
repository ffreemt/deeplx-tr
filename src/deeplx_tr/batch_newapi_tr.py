"""
Translate via newapi api.

also azure openai

"""
# pylint: disable=too-many-statements, too-many-branches, too-many-arguments

import asyncio
import datetime
from collections import defaultdict, deque
from contextlib import suppress
from pathlib import Path
from random import randrange
from time import monotonic
from typing import List

import diskcache
import yaml
from loadtext import loadtext
from loguru import logger

# from openai import OpenAI, AzureOpenAI
from ycecream import y

# from deeplx_tr.newapi_models import newapi_models
from deeplx_tr.newapi_tr import newapi_tr

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
# MODELS = newapi_models[:]

# DEQ = deque(newapi_models)

newapi_models = yaml.load(Path('model-list.yml').read_text(), Loader=yaml.Loader).get('models')
DEQ = deque(newapi_models)

y(DEQ, len(DEQ))

# initialize stats for each model, some are 3-tuple
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


async def worker(
    queue_texts,
    deq_models,
    n_items: int,
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
        n_items: total number of items in the queue
        wid: identifier of this worker for debugging and info collecting, default randrange(1000)
        queue_trtexts: async queue for translated texts, shared among possible many workers
        timeout: seconds per item in queue_texts, to exit While True loop
            to prevent hanging, default 30
        model_suffix: attache model name as suffix is True

    """
    if wid < 0:
        wid = randrange(1000)
    if timeout < 0:
        timeout = 30

    # n_times: fixed, expected queue_trtexts length
    # so that workers do not quit too early
    # n_items = queue_texts.qsize()

    logger.trace(f"{n_items=}, {wid=}")
    logger.trace(f"{queue_trtexts=}, {wid=}")

    then = monotonic()
    trtext_list = []
    idx = -1
    while True:
        idx += 1
        logger.trace(f"\n\t ########## {idx=}, {wid=}")
        if queue_trtexts.qsize() >= n_items or monotonic() - then > timeout * n_items:
            logger.trace("break on queue_trtexts.qsize() >= n_items or monotonic() - then > timeout * n_items")
            logger.trace(queue_trtexts.qsize() >= n_items)
            logger.trace(f"{queue_trtexts.qsize()=}")
            logger.trace(f"{n_items=}")
            logger.trace(monotonic() - then > timeout * n_items)

            break
        try:
            seqno_text = queue_texts.get_nowait()
            logger.trace(f"{seqno_text=}")

            # there is no need for this 'if', but just to play safe
            if len(seqno_text) == 2:
                seqno, text = seqno_text
                logger.trace(f"{seqno=}, {text=}")
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
        model_or_3_tuple = deq_models[-1]

        logger.trace(model_or_3_tuple)

        # just model
        if isinstance(model_or_3_tuple, str):
            model = model_or_3_tuple
            base_url, api_key = "", ""

        else:  # 3-tople: model, base_url, api_key
            model, base_url, api_key = model_or_3_tuple

        deq_models.rotate()
        logger.trace(f" deq_models rotated: {deq_models=}")
        logger.trace(f" {model=}")
        logger.trace(f" {text=}")

        # translate using async newapi_tr
        try:
            logger.trace(f" try newapi_tr {text=} {wid=}")
            # trtext = await newapi_tr(text, model=model)
            trtext = await newapi_tr(text, model=model, base_url=base_url, api_key=api_key)

            logger.trace(f" done newapi_tr {text=} {wid=}")
        except Exception as exc:  # pylint: disable=broad-except
            logger.trace(f"{exc=}, {wid=}")
            trtext = exc  # for retry in the subsequent round

            # optionally, remove model from deq_models since it does not deliver
            _ = """
            try:
                # deq_models.remove(model_or_3_tuple)
                ...
            except:  # pylint: disable=broad-except,bare-except  # noqa
                # maybe another worker already did deq_models.remove(model_or_3_tuple)
                ...
            """
        except:  # to make pyright happy  # noqa  # pylint: disable=bare-except
            # ...
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

                model_use_stats[model_or_3_tuple]["fail"] += 1

                await asyncio.sleep(0.01)  # give other workers a chance to try
            else:
                # text not empty but text.strip() empty, try gain
                if text.strip() and not trtext.strip():
                    logger.trace(
                        f"{wid=} {seqno=} empty trtext, back to the queue_texts"
                    )
                    # try again if trtext empty
                    await queue_texts.put((seqno, text))
                    await cache_incr("workers_emp", wid)

                    model_use_stats[model_or_3_tuple]["empty"] += 1

                    await asyncio.sleep(0.1)
                else:  # success
                    logger.info(f"{wid=} {seqno=} done ")

                    if model_suffix:
                        trtext = f"{trtext} [{model}]"

                    trtext_list.append((seqno, trtext))

                    await queue_trtexts.put((seqno, trtext))
                    await cache_incr("workers_succ", wid)

                    model_use_stats[model_or_3_tuple]["succ"] += 1
                    await asyncio.sleep(0.1)

    logger.trace(f"\n\t {trtext_list=}, {wid=} fini")

    return trtext_list


async def batch_newapi_tr(
    texts: List[str], n_workers: int = 4, model_suffix: bool = True
):
    """
    Translate in batch using urls from deq.

    Args:
    ----
        texts: list of text to translate
        n_workers: number of workers
        model_suffix: attache [model name] as suffix in the translated text if True

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

    # cap to len(texts) + 1
    n_workers = min(n_workers, len(texts)) + 1

    logger.info(f"{n_workers=}")

    que = asyncio.Queue()
    for idx, text in enumerate(texts):
        await que.put((idx, text))  # attach seq no for retry

    logger.trace(f"{que=}")

    cache.set("workers_succ", [0] * n_workers)
    cache.set("workers_fail", [0] * n_workers)
    cache.set("workers_emp", [0] * n_workers)

    queue_trtexts = asyncio.Queue()  # this seems necessary
    _ = """
    tasks = [
        asyncio.create_task(worker(que, DEQ, len(texts), wid_, model_suffix=model_suffix, queue_trtexts=queue_trtexts))
        for wid_ in range(n_workers)
    ]
    # """
    tasks = []
    for wid_ in range(n_workers):
        task = asyncio.create_task(worker(que, DEQ, len(texts), wid_, model_suffix=model_suffix, queue_trtexts=queue_trtexts))
        await asyncio.sleep(0)
        tasks.append(task)

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

    then_ = monotonic()
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

    print(f"{monotonic() - then_ :.2f}s")


if __name__ == "__main__":
    # asyncio.run(check())
    # asyncio.run(main())

    # texts = loadtext(r"C:\syncthing\00xfer\2021it\2024-05-30.txt", splitlines=1)
    texts_list = loadtext("tests/2024-06-20.txt")
    texts_list = texts_list[:30]

    # texts_list = loadtext(r"tests/test.txt")

    _ = """
    today = f"{datetime.date.today()}"
    with suppress(Exception):
        texts_list = loadtext(rf"C:\syncthing\00xfer\2021it\{today}.txt", splitlines=1)
    # """

    # _ = asyncio.run(batch_newapi_tr(["test 123", "test abc "]))

    then_1 = monotonic()

    # take a peek at n_workers dequeue
    _ = len(DEQ)
    y(_)

    texts_list = asyncio.run(
        batch_newapi_tr(texts_list),
    )

    texts_list = sorted(texts_list, key=lambda x: x[0])

    print("\n\n".join([", ".join(map(str, elm)) for elm in texts_list]))

    print(f"total: {monotonic() - then_1 :.2f}s")

    _ = """
    stats = [
        [*map(lambda x: model_use_stats[model][x], ["succ", "fail", "empty"])]

    ]
    # """

    y("------stats------")
    stats = []
    for model_ in DEQ:
        _ = []
        for elm in ["succ", "fail", "empty"]:
            _.append(model_use_stats[model_][elm])
        stats.append(_)

    print(f"{'model':<28}:", ["success", "failure", "empty"])
    # y(f"{'model':<28}:", ["success", "failure", "empty"])
    # for model_, _ in zip(MODELS, stats):
    for model_, _ in zip(newapi_models, stats):
        # print(f"{model_:<28}:", _)
        print(_, model_)
