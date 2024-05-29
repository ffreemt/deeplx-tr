"""
Test async deeplx_client.

Use -s or --capture=no e.g., pytest -s test_foobar.py to show output
"""
import pytest
from deeplx_tr.deeplx_client_async import deeplx_client_async as cl
from loguru import logger

pytestmark = pytest.mark.asyncio

async def test_deeplx_client_async():
    """Test deeplx_client_async."""
    _ = await cl("Test")
    logger.info(_)
    assert "测" in _

    _ = await cl("Test", alternatives=1)
    logger.info(_)
    assert "试" in _
