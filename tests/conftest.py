# pylint: disable=unused-argument, missing-module-docstring, missing-function-docstring
import asyncio
import os  # noqa
import pytest
from time import sleep
from python_run_cmd import run_cmd  # noqa
from ycecream import y

y.configure(sln=1, st=1)


# _ = '''
def pytest_sessionfinish(session, exitstatus):
    """Fix hanging attempt."""
    # print("<Put here your synthesis code>")
    # print("taskkill /f /pid to rid of hanging "
    #    "for openai.completion.chat.cfreate problem")
    # ugly hack, pytest wont print "n passed" etc
    # y(f"taskkill /f /pid {os.getpid()}")
    # run_cmd(f"taskkill /f /pid {os.getpid()}")
# '''

_ = """
@pytest.fixture(scope="session", autouse=True)
def configure_event_loop():
    policy = asyncio.get_event_loop_policy()
    policy.set_event_loop(policy.new_event_loop())

    yield

    loop = policy.get_event_loop()
    loop.close()

# """

# https://stackoverflow.com/questions/61022713/pytest-asyncio-has-a-closed-event-loop-but-only-when-running-all-tests
_= """
@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()

    # sleep(5)
    # ugly hack, pytest wont print "n passed" etc
    # run_cmd(f"taskkill /f /pid {os.getpid()}")
# """