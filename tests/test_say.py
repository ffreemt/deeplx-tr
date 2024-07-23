# https://readmedium.com/testing-asyncio-python-code-with-pytest-a2f3628f82bc

import asyncio
import pytest

# from say import say

async def say(what, when):
    await asyncio.sleep(when)
    return what


@pytest.mark.asyncio
async def test_say0():
    assert 'Hello0' == await say('Hello0', 0)


# @pytest.fixture  # deprecated, use @pytest.mark.asyncio
# def event_loop():
    # loop = asyncio.get_event_loop()
    # yield loop
    # loop.close()


async def test_say():
    assert 'Hello!' == await say('Hello!', 0)

async def test_say1():
    assert 'Hello' == await say('Hello', 0)

# _ = """
def test_say2(event_loop):
    assert 'Hello2' == event_loop.run_until_complete(say('Hello2', 0))

def test_say3(event_loop):
    # assert 'Hello3' == event_loop.run_until_complete(say('Hello3', 0))
    assert 'Hello3' == asyncio.run(say('Hello3', 0))
# """