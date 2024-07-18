# deeplx-tr
[![pytest](https://github.com/ffreemt/deeplx-tr/actions/workflows/routine-tests.yml/badge.svg)](https://github.com/ffreemt/deeplx-tr/actions)[![python](https://img.shields.io/static/v1?label=python+&message=3.8%2B&color=blue)](https://www.python.org/downloads/)[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)[![PyPI version](https://badge.fury.io/py/deeplx_tr.svg)](https://badge.fury.io/py/deeplx_tr)

deeplx and llm translate tool in python (wip)

## To run llm-tool
* `install rye` (google for instructions for yourplatform)
* `git clone https://github.com/ffreemt/deeplx-tr && cd deeplx-tr`
* `rye sync`
* `cp example.env .env` and amend `.env`
* `rye run taipy llm_tool.py`

## To use the deeplx_tr client
### Install it

```shell
pip install deeplx_tr --pre
```

### Use it

#### from command line
```bash
deeplx-tr hello world
# 哈罗世界

deeplx-tr hello world -t de
# Hallo Welt

deeplx-tr --help
```
or
```bash
python -m deeplx_tr hello world
python -m deeplx_tr hello world -d
python -m deeplx_tr --help
```
#### from python
```python
from deeplx_tr import deeplx_tr

res = deeplx_tr("hello world")
print(res)
# 哈罗世界

res = deeplx_tr("hello world", to_lang="de")
print(res)
# Hallo Welt
```
N.B. `deeplx-tr` will likely spit out `too many requestes` if you call it too often before long. But it's sufficient for ordinary average daily translation.
If you have a higher demand, try `deeplx.org` for which we provided two clients for your convenience.

## clients to query a `deeplx` server (default `deepx.org`)
```
from deeplx_tr import deeplx_client, deeplx_client_async

res = deeplx_client("hello world")
print(res)
# '哈罗世界'

res = deeplx_client("hello world", target_lang="de")
print(res)
# 'Hallo Welt'

# if you host your own deeplx, for example, at `127.0.0.1:1188'
# res = deeplx_client("hello world", url="http://127.0.0.1:1188/translate")
```

An async client is also available, e.g.
```python
import asyncio
from deeplx_tr import deeplx_client_async

async def main():
  res = await asyncio.gather(deeplx_client_async("hello world"), deeplx_client_async("test"))
  print(res)

asyncio.run(main())
# ['哈罗世界', '测试']
```
The default concurrency limit is `5` but can be altered by setting the environ variable CONCURRENCY_LIMIT, e.g.
```
set CONCURRENCY_LIMIT=8  # in Windows

# export CONCURRENCY_LIMIT=8 in Linux
```
