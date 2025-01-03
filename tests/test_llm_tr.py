"""Test llm_tr."""

# this seems to hang the test
import os  # noqa

# import sys
import pytest
from deeplx_tr.llm_tr import llm_tr
from python_run_cmd import run_cmd  # noqa
from ycecream import y

y.configure(sln=1)

text = "Test this and that."


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_llm_tr1_simple():
    """
    Test llm_tr simple.

    rye run pytest -s -k llm_tr_simple
    """
    trtext = await llm_tr(text)

    y(trtext)

    assert any(elm in str(trtext) for elm in ["测", "个", "翻译"])


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_llm_tr_gimini_2_flash_exp():
    """
    Test llm_tr gimini_2_flash_exp.

    rye run pytest -s -k llm_tr_gimini_2_flash_exp
    """
    trtext = await llm_tr(text, llm_model="gemini-2.0-flash-exp")

    y(trtext)

    assert any(elm in str(trtext) for elm in ["测", "个", "翻译"])


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_llm_tr1_d2a_cloudrun():
    """
    Test test_llm_tr_d2a_cloudrun.

    rye run pytest -s -k llm_tr_simple
    """
    base_url = "https://duck2api-service-2nsfpsd6ca-uc.a.run.app/v1"
    # base_url = "https://mikeee-duck2api.hf.space/hf/v1"
    trtext = await llm_tr(text, base_url=base_url, api_key="any")

    y(trtext)

    assert any(elm in str(trtext) for elm in ["测", "个", "翻译"])


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_llm_tr1_d2a_hf():
    """
    Test test_llm_tr_d2a_hf.

    rye run pytest -s -k llm_tr_simple
    """
    base_url = "https://duck2api-service-2nsfpsd6ca-uc.a.run.app"
    base_url = "https://mikeee-duck2api.hf.space/hf/v1"
    trtext = await llm_tr(text, base_url=base_url, api_key="any")

    y(trtext)

    assert any(elm in str(trtext) for elm in ["测", "个", "翻译"])
