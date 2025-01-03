r"""
Translate and chat via llm using marimo.

cf marimo-stuff\sidebar-ex-tabs.py
"""

import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import asyncio
    from itertools import zip_longest
    import io
    from pathlib import Path
    from time import sleep
    from types import SimpleNamespace

    import pandas as pd
    import numpy as np
    from unsync import unsync

    import marimo as mo

    # from ycecream import y

    from deeplx_tr.batch_deeplx_tr import batch_deeplx_tr
    from deeplx_tr.batch_newapi_tr import batch_newapi_tr
    from deeplx_tr.trtext2docx import trtext2docx
    return (
        Path,
        SimpleNamespace,
        asyncio,
        batch_deeplx_tr,
        batch_newapi_tr,
        io,
        mo,
        np,
        pd,
        sleep,
        trtext2docx,
        unsync,
        zip_longest,
    )


@app.cell
def _(
    Path,
    SimpleNamespace,
    batch_deeplx_tr,
    batch_newapi_tr,
    io,
    mo,
    pd,
    trtext2docx,
    unsync,
    zip_longest,
):
    # await asyncio.sleep(0)
    ns = SimpleNamespace()
    columns=['text', 'dxtext', 'lmtext']
    fileobj = mo.ui.file(filetypes=[".txt", ".csv"])

    def gen_dxtext():
        for _ in mo.status.progress_bar(
            range(1),
            title="diggin dxtr...",
            completion_title="done dxtr",
            # subtitle="wait...",
            # show_eta=True,
            # show_rate=True,
            # remove_on_exit=True,
        ):
            # await asyncio.sleep(3.5)
            # sleep(3.5)
            # asyncio.create_task(asyncio.sleep(3.5))
            try:
                # ns.dxtext = asyncio.create_task(batch_deeplx_tr(ns.text))
                _ = unsync(batch_deeplx_tr)(ns.text)
                _ = dict(_.result())
            except Exception as e:
                _ = [(0, str(e))]

            ns.dxtext = [dict(_).get(i) for i in range(len(_))]

        # ns.dxtext = ns.text[:]

        if not hasattr(ns, 'lmtext'):
            ns.lmtext = ['']
        ns.filename = fileobj.name()
        ns.docx_file = trtext2docx(ns.text, ns.dxtext, ns.lmtext)
        _ = io.BytesIO()
        trtext2docx(ns.text, ns.dxtext, ns.lmtext).save(_)
        ns.docx_file = _.getvalue()
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
            # sleep(3.5)
            try:
                _ = unsync(batch_newapi_tr)(ns.text)
                _ = dict(_.result())
            except Exception as e:
                _ = [(0, str(e))]

            ns.lmtext = [dict(_).get(i) for i in range(len(_))]

        # ns.lmtext = ns.text[:]

        if not hasattr(ns, 'dxtext'):
            ns.dxtext = ['']
        ns.filename = fileobj.name()

        # write to io.BytesIO and convert to bytes
        _ = io.BytesIO()
        trtext2docx(ns.text, ns.dxtext, ns.lmtext).save(_)
        ns.docx_file = _.getvalue()
        ns.df = pd.DataFrame([*zip_longest(ns.text, ns.dxtext, ns.lmtext, fillvalue='')], columns=columns)


    ns.text = ['']
    ns.dxtext = ['']
    ns.lmtext = ['']
    ns.docx_file = Path(r'C:\Users\User\Documents\111.docx').read_bytes()
    ns.filename = 'temp-marimo.txt'
    ns.df = pd.DataFrame([*zip_longest(ns.text, ns.dxtext, ns.lmtext, fillvalue='')], columns=columns)
    return columns, fileobj, gen_dxtext, gen_lmtext, ns


@app.cell
def _(
    Path,
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
    if fileobj.contents():
        lines = []
        try:
            lines = fileobj.contents().decode().splitlines()
        except Exception as e:
            y(e)
            raise
        ns.filename = fileobj.name()
        ns.text = [elm.strip() for elm in lines]
        ns.df = pd.DataFrame([*zip_longest(ns.text, ns.dxtext, ns.lmtext, fillvalue='')], columns=columns)

    # ns.docx_file = Path(r'C:\Users\User\Documents\111.docx').read_bytes()
    # _ = f'{Path(ns.filename).stem}-tr.docx'

    button_dl = mo.download(
        data=ns.docx_file,
        filename=f'{Path(ns.filename).stem}-tr.docx',
        # mimetype="text/plain",
        # mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        label="dl docx",
    )

    file_tab = mo.vstack([
        # mo.hstack([fileobj]),
        mo.hstack(
            [fileobj, button_dx, button_lm, button_dl,],
            align='start',
            widths=[4, 1, 1, 1,]
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
    # file_tab

    return button_dl, file_tab, lines


@app.cell
def _(mo):
    settings = mo.vstack(
        [
            mo.md("Edit User"),
            first := mo.ui.text(label="First Name"),
            last := mo.ui.text(label="Last Name"),
        ]
    )

    organization = mo.vstack(
        [
            mo.md("Edit Organization"),
            org := mo.ui.text(label="Organization Name", value="..."),
            employees := mo.ui.number(
                label="Number of Employees", start=0, stop=1000
            ),
        ]
    )

    info = mo.vstack(
        [
            # mo.md("## Info"),
            mo.md(
                """* 🐧 Join qqgroup 316287378 for updates and/or realtime chat.
                * ֎🇦🇮 If you have a  linux.do account, you may wish to check 
                this out: 
                [https://horizon.dattw.eu.org](https://horizon.dattw.eu.org) 
                (a free service courtesy of yours sincerely)"""
            ),
        ]
    )

    mo.sidebar(
        [
            # mo.md("## llmtool"),
            mo.nav_menu(
                {
                    "#/": f"{mo.icon('lucide:file')} File",
                    "#/llmtool": f"{mo.icon('lucide:bot')} Llmtool",
                    "#/info": f"{mo.icon('lucide:info')} Info",
                    # "Links": {
                    #     "https://twitter.com/marimo_io": "Twitter",
                    #     "https://github.com/marimo-team/marimo": "GitHub",
                    # },
                },
                orientation="vertical",
            ),
        ]
    )
    return employees, first, info, last, org, organization, settings


@app.cell
def _(file_tab, info, mo, organization):
    mo.routes({
        # "#/": mo.md("# Home"),
        # "#/": settings,
        "#/": file_tab,
        "#/llmtool": organization,
        "#/info": info,
        mo.routes.CATCH_ALL: file_tab,
    })
    return


@app.cell
def _(gen_dxtext, gen_lmtext, mo):

    button_dx = mo.ui.run_button(
        label='dxtr',
        kind='success',
        tooltip='click to translate via deeplx',
        on_change=lambda _: gen_dxtext(),
    )

    button_lm = mo.ui.run_button(
        label='lmtr',
        kind='success',
        tooltip='click to translate via llm',
        on_change=lambda _: gen_lmtext()
    )

    return button_dx, button_lm


if __name__ == "__main__":
    app.run()
