"""Test newapi_tr."""

# this seems to hang the test
import os  # noqa

import pytest
from deeplx_tr.newapi_tr import newapi_tr

# from deeplx_tr.newapi_tr_langchain import newapi_tr_langchain as newapi_tr
from python_run_cmd import run_cmd  # noqa
from ycecream import y

y.configure(sln=1)

text = "Test this and that."

# gpt-3.5-turbo
_ = """
@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
"""


def teardown_module1():
    print("module end here 1")
    print("module end here 2")
    print("module end here 3")


@pytest.fixture(scope="module", autouse=True)
def kill_self():
    yield

    # sleep(3)
    # print(" kill self")
    # run_cmd(f"taskkill /f /pid {os.getpid()}")
    print(" module end ")


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_newapi_tr2_langchain_simple():
    """
    Test newapi_tr simple.

    rye run pytest -s -k newapi_tr_simple
    """
    # asyncio.set_event_loop(event_loop)

    trtext = await newapi_tr(text)

    y(trtext)

    # assert any(elm in str(trtext) for elm in ["测", "个", "翻译"])
    assert any(elm in trtext for elm in ["测", "个"])


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_newapi_tr2_langchain_d2a_cloudrun():
    """
    Test test_newapi_tr_d2a_cloudrun.

    rye run pytest -s -k newapi_tr_simple
    """
    # asyncio.set_event_loop(event_loop)

    base_url = "https://duck2api-service-2nsfpsd6ca-uc.a.run.app/v1"
    # base_url = "https://mikeee-duck2api.hf.space/hf/v1"
    trtext = await newapi_tr(text, base_url=base_url, api_key="any")

    y(trtext)

    # assert any(elm in str(trtext) for elm in ["测", "个", "翻译"])
    assert any(elm in trtext for elm in ["测", "个"])


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_newapi_tr2_langchain_d2a_hf():
    """
    Test test_newapi_tr_d2a_hf.

    rye run pytest -s -k newapi_tr_simple
    """
    # asyncio.set_event_loop(event_loop)

    base_url = "https://duck2api-service-2nsfpsd6ca-uc.a.run.app"
    base_url = "https://mikeee-duck2api.hf.space/hf/v1"
    trtext = await newapi_tr(text, base_url=base_url, api_key="any")

    y(trtext)

    # assert any(elm in str(trtext) for elm in ["测", "个", "翻译"])
    assert any(elm in trtext for elm in ["测", "个"])
