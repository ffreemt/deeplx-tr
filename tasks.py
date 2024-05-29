"""
Define invoke tasks.

invoke list
invoke build

invoke --help build

"""
from invoke import task

name_def = "xxx"

@task(
    # default=True,
    help={'name': "Name of the person to say hi to."},
)
def build(c, name=name_def):
# def build(c):
    """
    Build tests.

    More explanations
    """
    c.run(f"echo sphinx-build docs docs/_build {name}")
    # c.run(f"echo sphinx-build docs docs/_build ")

@task(
    default=True,
    # help={'name': "Name of the person to say hi to."},
)
def run_batch_tr(c):
    """rye run python run-quick-freegpt-art.py."""
    c.run("rye run python run-batch_tr.py")
