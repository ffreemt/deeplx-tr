"""
Translate using deeplx-sites from cache.get("deeplx-sites").

cache = diskcache.Cache(Path.home() / ".diskcache" / "deeplx-sites")

# reverse, prepare for deq[-1] and deq.rotate
deq = deque([url for url, deplay in cache.get("deeplx-sites")[::-1]]
"""
import asyncio
from collections import deque
from pathlib import Path
from typing import List, Union

import diskcache
from loguru import logger
from ycecream import y

from deeplx_tr.deeplx_client_async import deeplx_client_async

cache = diskcache.Cache(Path.home() / ".diskcache" / "deeplx-sites")
_ = cache.get("deeplx-sites")
DEQ = deque([url for url, delay in _[::-1]])  # type: ignore


async def cache_incr(item, idx, inc=1):
    """Increase cache.get(item)[idx] += inc."""
    _ = cache.get(item)
    # silently ignore all exceptions
    try:
        # _[idx] = _[idx] + inc
        _[idx] += inc
    except Exception:
        return

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, cache.set, item, _)


async def worker(queue, deq, id=0) -> List[List[Union[str, BaseException]]]:
    """
    Translate text in the queue.

    Args:
    ----
    queue: asyncio.Queue that contains list of texts
    deq: collections.deque to hold deeplx urls

    url: deeplx site's url from a deque

    """
    logger.trace(f"******** {id=}")
    res = []

    # while not queue.empty():
    # try n times and break
    n_attempts = queue.qsize()

    logger.trace(f"{n_attempts=} {id=}")

    for _ in range(n_attempts):
        logger.trace(f"attemp {_ + 1}  {id=} ")
        if queue.empty():
            logger.trace(f" queue empty, done {id=} ")
            break

        try:
            seqno_text = queue.get_nowait()
            logger.trace(f"{seqno_text=}")
            if len(seqno_text) == 2:
                seqno, text = seqno_text
            else:
                text = str(seqno_text)
                seqno = -1
        # another worker maybe manages to empty the queue in the mean time
        except asyncio.QueueEmpty as exc:
            logger.warning(f"This should not happen, unless there is a race, {exc=}")
            text = exc
            break
        except Exception as exc:
            logger.warning(f"{exc=}")
            raise

        # process output (text) from queue.get_nowait()
        if not isinstance(text, asyncio.QueueEmpty):
            # fetch an url from deq's end
            # do we need to lock deq properly?
            url = deq[-1]
            deq.rotate()
            logger.trace(f" deq rotated: {deq=}")
            logger.trace(f" {url=}")
            logger.trace(f" {text=}")

            # httpx.HTTPStatusError
            try:
                logger.trace(f" try deeplx_client_async {text=} {id=}")
                trtext = await deeplx_client_async(text, url=url)
                logger.trace(f" done deeplx_client_async {text=} {id=}")
            except Exception as exc:
                logger.trace(f"{exc=}, {id=}")
                # raise
                trtext = exc  # for retry in the subsequent round

                # remove url from DEQ since it does not deliver
                try:
                    # DEQ.remove(url)
                    ...
                except Exception: # maybe another worker already did
                    ...
            finally:
                logger.trace(f" {trtext=}  {id=} ")
                queue.task_done()
                logger.trace(f"\n\t >>====== que.task_done()  {id=}")

                # put text back in the queue if Exception
                if isinstance(trtext, Exception):
                    logger.info(f" {seqno=} failed {trtext=}, back to the queue")
                    await queue.put((seqno, text))
                    await cache_incr("workers_fail", id)
                else:
                    # text not empty but text.strip() empty, try gain
                    if text.strip() and not trtext.strip():
                        logger.info(f" {seqno=} empty trtext, back to the queue")
                        # try again if trtext empty
                        await queue.put((seqno, text))
                        await cache_incr("workers_emp", id)
                    else:
                        logger.info(f" {seqno=} done ")
                        res.append((seqno, trtext))
                        await cache_incr("workers_succ", id)
    else:
        logger.trace(f" max attempts reached {id=}")

    logger.trace(f"\n\t {res=}, {id=} fini")

    return res


async def batch_tr(texts: List[str], n_workers: int = 20):
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
    # logger.trace(f"{texts=}")
    # logger.trace(y(texts))

    try:
        n_workers = int(n_workers)
    except Exception:
        n_workers = 20
    if n_workers < 1:
        n_workers = 20

    logger.debug(y(n_workers))

    que = asyncio.Queue()
    for idx, text in enumerate(texts):
        await que.put((idx, text))  # attach seq no for retry

    logger.trace(f"{que=}")

    # n_workers = 2
    # n_workers = 20
    # coros = [worker(que, DEQ, _) for _ in range(n_workers)]

    # does not run, must wrap in with asyncio.create_task
    # tasks = [worker(que, DEQ, _) for _ in range(n_workers)]

    # collect stats about workers
    # cache.set('workers_succ')
    # cache.set('workers_fail')
    # cache.set('workers_emp')
    cache.set("workers_succ", [0] * n_workers)
    cache.set("workers_fail", [0] * n_workers)
    cache.set("workers_emp", [0] * n_workers)

    tasks = [asyncio.create_task(worker(que, DEQ, _)) for _ in range(n_workers)]

    # needed
    await que.join()  # queue.task_done() for each task to properly exit

    logger.trace("\n\t  >>>>>>>> after  await que.join()")

    # give the last task some time
    await asyncio.sleep(.1)

    # Cancel our worker tasks, do we need this?
    # for task in tasks: task.cancel()

    logger.trace("\n\t >>>>>>>> Start await asyncio.gather")

    # consume texts_list in an async way
    # res = await asyncio.gather(*coros, return_exceptions=True)

    # res = await asyncio.gather(*tasks, return_exceptions=True)
    res = await asyncio.gather(*tasks)

    logger.trace("\n\t  >>>>>>>> Done await asyncio.gather")

    # print(res[:3])

    logger.trace(f"{res=}")
    # return res

    # res can be asyncio.CancelledError in which case

    res1 = []
    for _ in res:
        res1.extend(_)  # type: ignore
    logger.trace(f"{res1=}")

    succ = cache.get("workers_succ")
    logger.info(f"""success\n\t {succ}, {sum(succ)}""")
    logger.info(f"""failure\n\t {cache.get("workers_fail")}""")
    logger.info(f"""empty\n\t {cache.get("workers_emp")}""")

    return res1


if __name__ == "__main__":
    _ = asyncio.run(batch_tr(['test 123', 'test abc ']))
    print(_)
