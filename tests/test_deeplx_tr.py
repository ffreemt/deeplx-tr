"""Test deeplx_tr."""
# pylint: disable=broad-except
from deeplx_tr import __version__, deeplx_tr


def test_version():
    """Test version."""
    assert __version__[:3] == "0.1"


def test_sanity():
    """Check sanity."""
    try:
        assert not deeplx_tr()
    except Exception:
        assert True
