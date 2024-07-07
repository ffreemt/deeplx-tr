# pylint: disable=unused-argument, missing-module-docstring, missing-function-docstring
import asyncio
import os  # noqa
import pytest
from python_run_cmd import run_cmd  # noqa

_ = '''
def pytest_sessionfinish(session, exitstatus):
    """Fix hanging attempt."""
    # print("<Put here your synthesis code>")
    # print("taskkill /f /pid to rid of hanging "
    #    "for openai.completion.chat.cfreate problem")
    # ugly hack, pytest wont print "n passed" etc
    # run_cmd(f"taskkill /f /pid {os.getpid()}")
# '''

@pytest.fixture(scope="session", autouse=True)
def configure_event_loop():
    policy = asyncio.get_event_loop_policy()
    policy.set_event_loop(policy.new_event_loop())

    yield

    loop = policy.get_event_loop()
    loop.close()