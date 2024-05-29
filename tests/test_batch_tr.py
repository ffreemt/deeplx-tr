"""
Test async batch_tr.

Use -s or --capture=no e.g., pytest -s test_foobar.py to show output
"""
import pytest
from deeplx_tr.batch_tr import batch_tr
from loguru import logger

pytestmark = pytest.mark.asyncio

async def test_batch_tr2():
    """Test batch_tr."""
    _ = await batch_tr(["Test 123", "test abc"])
    logger.info(_)
    assert "测" in _

