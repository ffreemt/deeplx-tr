# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.10.7"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Tables

        > “Sometimes I’ll start a sentence and I don’t even know where it’s going. I just hope I find it along the way.”
        — Michael Scott
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""_Create rich tables with selectable rows using_ `mo.ui.table`.""")
    return


@app.cell
def _():
    from types import SimpleNamespace

    ns = SimpleNamespace()

    # get_states, set_states = mo.state(ns)
    return SimpleNamespace, ns


@app.cell
def _():
    import asyncio
    from time import sleep
    import pandas as pd
    import numpy as np
    from ycecream import y

    y.configure(sln=1)
    columns=['text', 'dxtext', 'lmtext']

    return asyncio, columns, np, pd, sleep, y


@app.cell
def _(mo):
    fileobj = mo.ui.file(filetypes=[".txt", ".csv"])

    # print(f'{fileobj=}, {type(fileobj)=}')
    # g"{fileobj.contents=}"

    return (fileobj,)


@app.cell
def _(asyncio, columns, mo, ns, pd, sleep):
    from itertools import zip_longest

    async def gen_dxtext():
        for _ in mo.status.progress_bar(
            range(1),
            title="diggin dxtr...",
            completion_title="done dxtr",
            # subtitle="wait...",
            # show_eta=True,
            # show_rate=True,
            # remove_on_exit=True,
        ):
            await asyncio.sleep(3.5)
            # sleep(3.5)

        ns.dxtext = ns.text[:]
        if not hasattr(ns, 'lmtext'):
            ns.lmtext = ['']
        ns.df = pd.DataFrame([*zip_longest(ns.text, ns.dxtext, ns.lmtext, fillvalue='')], columns=columns)

    def gen_lmtext():
        for _ in mo.status.progress_bar(
            range(1),
            title="diggin lmtr ...",
            completion_title="done lmtr",
            # subtitle="wait...",
            # show_eta=True,
            # show_rate=True,
            # remove_on_exit=True,
        ):
            # await asyncio.sleep(3.5)
            sleep(3.5)

        ns.lmtext = ns.text[:]
        if not hasattr(ns, 'dxtext'):
            ns.dxtext = ['']
        ns.df = pd.DataFrame([*zip_longest(ns.text, ns.dxtext, ns.lmtext, fillvalue='')], columns=columns)


    ns.text = ['']
    ns.dxtext = ['']
    ns.lmtext = ['']
    ns.df = pd.DataFrame([*zip_longest(ns.text, ns.dxtext, ns.lmtext, fillvalue='')], columns=columns)

    # print(ns)

    # y(ns.table1)
    # ns.table1
    return gen_dxtext, gen_lmtext, zip_longest


@app.cell
def _():
    return


@app.cell
def file_menu_(
    button_dl,
    button_dx,
    button_lm,
    columns,
    fileobj,
    mo,
    ns,
    pd,
    y,
    zip_longest,
):

    # table
    if fileobj.contents():
        lines = []
        try:
            lines = fileobj.contents().decode().splitlines()
        except Exception as e:
            y(e)
            raise
        ns.text = [elm.strip() for elm in lines]
        ns.df = pd.DataFrame([*zip_longest(ns.text, ns.dxtext, ns.lmtext, fillvalue='')], columns=columns)

    file_menu = mo.vstack([
        # mo.hstack([fileobj]),
        mo.hstack(
            [fileobj, button_dx, button_lm, button_dl],
            align='start',
            widths=[4, 1, 1, 1]
        ),
        mo.accordion({
            "...": mo.ui.table(
                [dict(zip(ns.df.columns, elm)) for elm in ns.df.values.tolist()],
                # _,
                # show_column_summaries=None,
                # text_justify_columns={'text': 'left'},
                selection=None,  # 'single',  # 'None,
                wrapped_columns=['text', 'dxtext', 'lmtext'],
                page_size=15,
                # label='llm text'
            )
        }),
    ])
    file_menu
    return (lines, file_menu)


file_menu = file_menu_()[1]


@app.cell
async def _(asyncio, gen_dxtext, gen_lmtext, mo):
    # need this cell be async
    await asyncio.sleep(0)

    def button_handler(_):
        async def async_handler():
            try:
                result = await gen_dxtext()
                # mo.ui.text(result).update()
            except Exception as e:
                mo.ui.text(f"Error: {str(e)}").update()

        # Create and run the task
        asyncio.create_task(async_handler())

    # def button_handler(_):
    #     print(_)
    #     gen_dxtext()

    button_dx = mo.ui.run_button(
        label='dxtr',
        kind='success',
        tooltip='cilck to translate via deeplx',
        # on_change=lambda _: gen_dxtext(),
        on_change=button_handler,
    )

    button_lm = mo.ui.run_button(
        label='lmtr',
        kind='success',
        tooltip='cilck to translate via llm',
        on_change=lambda _: gen_lmtext()
    )
    return button_dx, button_handler, button_lm


@app.cell
def _(mo):
    from pathlib import Path
    docx_file = Path(rf'C:\Users\User\Documents\111.docx').read_bytes()
    button_dl = mo.download(
        data=docx_file,
        filename="111hello.docx",
        # mimetype="text/plain",
        mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        label="dl docx",
    )
    return Path, button_dl, docx_file


@app.cell
async def _(asyncio, mo, rerun):

    for _ in mo.status.progress_bar(
        range(1),
        title="Loading",
        subtitle="Please wait",
        # show_eta=True,
        # show_rate=True,
        remove_on_exit=True,
    ):
        await asyncio.sleep(3.5)
    rerun
    return


@app.cell
def _(mo):
    rerun = mo.ui.button(label="Rerun")
    rerun
    return (rerun,)


@app.cell
def _(mo):
    mo.md("""**Single selection.**""")
    return


@app.cell
def _(mo, office_characters):
    single_select_table = mo.ui.table(
        office_characters,
        selection="single",
        pagination=True,
    )
    return (single_select_table,)


@app.cell
def _(mo, single_select_table):
    mo.ui.tabs({"table": single_select_table, "selection": single_select_table.value})
    return


@app.cell
def _(mo):
    mo.md("""**Multi-selection.**""")
    return


@app.cell
def _(mo, office_characters):
    multi_select_table = mo.ui.table(
        office_characters,
        selection="multi",
        pagination=True,
    )
    return (multi_select_table,)


@app.cell
def _(mo, multi_select_table):
    mo.ui.tabs({"table": multi_select_table, "selection": multi_select_table.value})
    return


@app.cell
def _(mo):
    mo.md("""**No selection.**""")
    return


@app.cell
def _(mo, office_characters):
    table = mo.ui.table(
        office_characters,
        label="Employees",
        selection=None,
    )

    table
    return (table,)


@app.cell
def _(mo):
    office_characters = [
        {
            "first_name": "Michael",
            "last_name": "Scott",
            "skill": mo.ui.slider(1, 10, value=3),
            "favorite place": mo.image(src="https://picsum.photos/100", rounded=True),
        },
        {
            "first_name": "Jim",
            "last_name": "Halpert",
            "skill": mo.ui.slider(1, 10, value=7),
            "favorite place": mo.image(src="https://picsum.photos/100"),
        },
        {
            "first_name": "Pam",
            "last_name": "Beesly",
            "skill": mo.ui.slider(1, 10, value=3),
            "favorite place": mo.image(src="https://picsum.photos/100"),
        },
        {
            "first_name": "Dwight",
            "last_name": "Schrute",
            "skill": mo.ui.slider(1, 10, value=7),
            "favorite place": mo.image(src="https://picsum.photos/100"),
        },
        {
            "first_name": "Angela",
            "last_name": "Martin",
            "skill": mo.ui.slider(1, 10, value=5),
            "favorite place": mo.image(src="https://picsum.photos/100"),
        },
        {
            "first_name": "Kevin",
            "last_name": "Malone",
            "skill": mo.ui.slider(1, 10, value=3),
            "favorite place": mo.image(src="https://picsum.photos/100"),
        },
        {
            "first_name": "Oscar",
            "last_name": "Martinez",
            "skill": mo.ui.slider(1, 10, value=3),
            "favorite place": mo.image(src="https://picsum.photos/100"),
        },
        {
            "first_name": "Stanley",
            "last_name": "Hudson",
            "skill": mo.ui.slider(1, 10, value=5),
            "favorite place": mo.image(src="https://picsum.photos/100"),
        },
        {
            "first_name": "Phyllis",
            "last_name": "Vance",
            "skill": mo.ui.slider(1, 10, value=5),
            "favorite place": mo.image(src="https://picsum.photos/100"),
        },
        {
            "first_name": "Meredith",
            "last_name": "Palmer",
            "skill": mo.ui.slider(1, 10, value=7),
            "favorite place": mo.image(src="https://picsum.photos/100"),
        },
        {
            "first_name": "Creed",
            "last_name": "Bratton",
            "skill": mo.ui.slider(1, 10, value=3),
            "favorite place": mo.image(src="https://picsum.photos/100"),
        },
        {
            "first_name": "Ryan",
            "last_name": "Howard",
            "skill": mo.ui.slider(1, 10, value=5),
            "favorite place": mo.image(src="https://picsum.photos/100"),
        },
        {
            "first_name": "Kelly",
            "last_name": "Kapoor",
            "skill": mo.ui.slider(1, 10, value=3),
            "favorite place": mo.image(src="https://picsum.photos/100"),
        },
        {
            "first_name": "Toby",
            "last_name": "Flenderson",
            "skill": mo.ui.slider(1, 10, value=3),
            "favorite place": mo.image(src="https://picsum.photos/100"),
        },
        {
            "first_name": "Darryl",
            "last_name": "Philbin",
            "skill": mo.ui.slider(1, 10, value=7),
            "favorite place": mo.image(src="https://picsum.photos/100"),
        },
        {
            "first_name": "Erin",
            "last_name": "Hannon",
            "skill": mo.ui.slider(1, 10, value=5),
            "favorite place": mo.image(src="https://picsum.photos/100"),
        },
        {
            "first_name": "Andy",
            "last_name": "Bernard",
            "skill": mo.ui.slider(1, 10, value=5),
            "favorite place": mo.image(src="https://picsum.photos/100"),
        },
        {
            "first_name": "Jan",
            "last_name": "Levinson",
            "skill": mo.ui.slider(1, 10, value=5),
            "favorite place": mo.image(src="https://picsum.photos/100"),
        },
        {
            "first_name": "David",
            "last_name": "Wallace",
            "skill": mo.ui.slider(1, 10, value=3),
            "favorite place": mo.image(src="https://picsum.photos/100"),
        },
        {
            "first_name": "Holly",
            "last_name": "Flax",
            "skill": mo.ui.slider(1, 10, value=7),
            "favorite place": mo.image(src="https://picsum.photos/100"),
        },
    ]
    return (office_characters,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
