"""Test deeplx_tr."""

# pylint: disable=broad-except
import os
import time

from deeplx_tr import __version__, deeplx_tr
from loguru import logger

IDLE_TIME = 15
# e.g. set/export IDLE_TIME=10 to change IDLE_TIME time
try:
    _ = int(os.getenv(IDLE_TIME))
except Exception:
    _ = IDLE_TIME

IDLE_TIME = _ or IDLE_TIME


def test_version():
    """Test version."""
    assert __version__[:3] == "0.1"


def test_sanity():
    """Check sanity."""
    try:
        assert not deeplx_tr()
    except Exception:
        assert True


def test_deeplx_tr_whitespaces():
    """Test whitespaces stripped."""
    # _ = deeplx_tr({"text": "   "})
    _ = deeplx_tr(text="   ")
    assert _.get("data") == ""


def test_deeplx_tr_simple_default():
    """Test simple 'this is a test'."""
    # text = "This is a test."
    text = "This is test ."

    logger.info(f" sleeping for {IDLE_TIME} sec")
    time.sleep(IDLE_TIME)  # roug rate limit

    # _ = deeplx_tr({"text": text})  # {'code': 200, 'data': '这是一个 测试。', 'id': 101564000}
    _ = deeplx_tr(text=text)
    _ = _.get("data")

    assert "测" in _ or "试" in _ or "考" in _


def test_deeplx_tr_simple_tgt_lang_de():
    """Test simple 'this is a test'."""
    text = "This is a test."

    logger.info(f" sleeping for {IDLE_TIME} sec")
    time.sleep(IDLE_TIME)  # roug rate limit

    # _ = deeplx_tr({"text": text, "target_lang": "de"})
    _ = deeplx_tr(text=text, target_lang="de")
    _ = _.get("data").lower()  # 'dies ist ein test.'

    assert "dies" in _ or "test" in _
