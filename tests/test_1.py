"""Test newapi_tr."""

import pytest

# from deeplx_tr.newapi_tr import newapi_tr
from deeplx_tr.newapi_tr_langchain import newapi_tr_langchain as newapi_tr

text = "Test this and that."


# @pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_1():
    # def test_1(event_loop):
    """
    Test newapi_tr simple.

    rye run pytest -s -k newapi_tr_simple
    """
    # asyncio.set_event_loop(event_loop)

    base_url = "https://mikeee-duck2api.hf.space/hf/v1"
    trtext = await newapi_tr(text, base_url=base_url, api_key="any")

    trtext = await newapi_tr(text)

    # trtext = event_loop.run_until_complete(newapi_tr(text))

    # assert any(elm in str(trtext) for elm in ["测", "个", "翻译"])
    assert any(elm in trtext for elm in ["测", "个"])
