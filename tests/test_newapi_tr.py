"""Test newapi_tr."""

# this seems to hang the test
import os  # noqa

# import sys
import pytest
from deeplx_tr.newapi_tr import newapi_tr
from python_run_cmd import run_cmd  # noqa
from ycecream import y

y.configure(sln=1)

text = "Test this and that."


def teardown_module1():
    print("module end here 1")
    print("module end here 2")
    print("module end here 3")


@pytest.fixture(scope="module", autouse=True)
def kill_self():
    yield

    from time import sleep

    sleep(3)
    # print(" kill self")
    # run_cmd(f"taskkill /f /pid {os.getpid()}")
    print(" tr1 module end ")


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_newapi_tr1_simple():
    """
    Test newapi_tr simple.

    rye run pytest -s -k newapi_tr_simple
    """
    trtext = await newapi_tr(text)

    y(trtext)

    assert any(elm in str(trtext) for elm in ["测", "个", "翻译"])


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_newapi_tr1_d2a_cloudrun():
    """
    Test test_newapi_tr_d2a_cloudrun.

    rye run pytest -s -k newapi_tr_simple
    """
    base_url = "https://duck2api-service-2nsfpsd6ca-uc.a.run.app/v1"
    # base_url = "https://mikeee-duck2api.hf.space/hf/v1"
    trtext = await newapi_tr(text, base_url=base_url, api_key="any")

    y(trtext)

    assert any(elm in str(trtext) for elm in ["测", "个", "翻译"])


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_newapi_tr1_d2a_hf():
    """
    Test test_newapi_tr_d2a_hf.

    rye run pytest -s -k newapi_tr_simple
    """
    base_url = "https://duck2api-service-2nsfpsd6ca-uc.a.run.app"
    base_url = "https://mikeee-duck2api.hf.space/hf/v1"
    trtext = await newapi_tr(text, base_url=base_url, api_key="any")

    y(trtext)

    assert any(elm in str(trtext) for elm in ["测", "个", "翻译"])
