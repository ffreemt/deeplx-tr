"""Init."""
__version__ = "0.1.0a0"
from .deeplx_tr import deeplx_tr, lang_list
from .deeplx_client import deeplx_client
from .deeplx_client_async import deeplx_client_async

__all__ = ("deeplx_tr", "deeplx_client", "deeplx_client_async", "lang_list")
