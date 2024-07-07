"""
Define invoke tasks.

inv -l
invoke --list
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
    # default=True,
)
def scrape_deeplx(c):
    """Scrape shodan/shoda ip."""
    # c.run(r"python src\deeplx_tr\scrape_deeplx_shodan.py")
    c.run("rye run python -m deeplx_tr.scrape_deeplx_shodan")


@task()
def trtext2docx(c):
    """Test trtext2docx."""
    # c.run("py trtext2docx")
    c.run("python -m deeplx_tr.trtext2docx")


@task()
def pytest1(c):
    """Test batch_newapi_tr simple."""
    # c.run("py newapi_tr.py")
    # c.run("rye run pytest -s -k newapi_tr_simple")
    c.run("rye run pytest -s -k newapi_tr1")

@task()
def pytest2(c):
    """Test batch_newapi_tr_langchain simple."""
    c.run("rye run pytest -s -k newapi_tr2_langchain")


@task()
def batch_newapi_tr(c):
    """Test batch_newapi_tr."""
    # c.run("py newapi_tr.py")
    c.run("python -m deeplx_tr.batch_newapi_tr")


@task(
    default=True,
)
def run_batch_deeplx_tr(c):
    """rye run python run-batch_tr.py (deeplx_tr.batch_tr)."""
    c.run("rye run python run_batch_deeplx_tr.py")
