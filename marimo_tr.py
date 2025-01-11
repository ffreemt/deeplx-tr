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

    from loguru import logger
    import pandas as pd
    import numpy as np
    from unsync import unsync

    import marimo as mo

    from ycecream import y
    y.configure(e=0)

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
        logger,
        mo,
        np,
        pd,
        sleep,
        trtext2docx,
        unsync,
        y,
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

    def reset_pbar():
        for _ in mo.status.progress_bar(
            range(1),
            title="remove pbar ...",
            # completion_title="done lmtr",
            remove_on_exit=True,
        ):
            ...
        ...

    ns.text = ['']
    ns.dxtext = ['']
    ns.lmtext = ['']
    ns.docx_file = Path('empty.docx').read_bytes()
    ns.filename = 'temp-marimo.txt'
    ns.df = pd.DataFrame([*zip_longest(ns.text, ns.dxtext, ns.lmtext, fillvalue='')], columns=columns)

    ns.row_tot = 1
    ns.row_numb = 0
    return columns, fileobj, gen_dxtext, gen_lmtext, ns, reset_pbar


@app.cell
def _(mo):
    get_state, set_state = mo.state(0)  # row_numb

    get_tot, set_tot = mo.state(1)  # total # of rows
    return get_state, get_tot, set_state, set_tot


@app.cell
def _(get_state, mo, ns):
    # for lmtool_tab
    row1 = mo.ui.text_area(value=ns.text[get_state()], rows=2, full_width=True)
    row2 = mo.ui.text_area(value=ns.dxtext[get_state()], rows=2,  full_width=True)
    row3 = mo.ui.text_area(value=ns.lmtext[get_state()], rows=2,  full_width=True)
    return row1, row2, row3


@app.cell
def _(logger, ns, row1, set_state):
    def set_row_numb(n):
        logger.info(f"{n=}")
        # logger.info(ns)
        ns.row_numb = n
        set_state(ns.row_numb)

        # row1 = ns.text[n]
        # row2 = ns.dxtext[n]
        # row3 = ns.lmtext[n]
        # logger.info(row1)

        # not allowed, have to use mo.state
        # row1.value = ns.text[n]
        # row2.value = ns.dxtext[n]
        # row3.value = ns.lmtext[n]

        logger.info(f"{row1.value=}")
    return (set_row_numb,)


@app.cell
def _(columns, fileobj, ns, pd, set_state, set_tot, y, zip_longest):
    _ = """ pandas.read_csv
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False)
     as temp_file:
        temp_file.write(bytes_c)
        temp_file_path = temp_file.name
    pd.read_csv(temp_file_path, header=None, index=None)

    """
    if fileobj.contents():
        lines = []
        try:
            lines = fileobj.contents().decode().splitlines()
        except Exception as e:
            y(e)
            raise
        ns.filename = fileobj.name()
        ns.text = [elm.strip() for elm in lines if elm.strip()]
        ns.df = pd.DataFrame([*zip_longest(ns.text, ns.dxtext, ns.lmtext, fillvalue='')], columns=columns)

        # update dxtext, lmtext
        _, ns.dxtext, ns.lmtext = ns.df.values.T.tolist() 
        ns.row_tot = len(ns.text)
        ns.row_numb = 0
        set_state(ns.row_numb)
        set_tot(ns.row_tot)
    return (lines,)


@app.cell
def _(
    Path,
    button_dx,
    button_lm,
    button_reset_pbar,
    fileobj,
    get_state,
    get_tot,
    logger,
    mo,
    ns,
    row1,
    row2,
    row3,
    set_row_numb,
):
    # _ = slider_row_numb.value
    slider_row_numb = mo.ui.slider(
        start=0,
        stop=ns.row_tot - 1,
        value=get_state(),
        step=1,
        label=f"row-{get_state()}/{get_tot()}",
        show_value=True,
        on_change=set_row_numb,
        # on_change=set_state,
        full_width=True,
    )

    # ns.docx_file = Path(r'C:\Users\User\Documents\111.docx').read_bytes()
    # _ = f'{Path(ns.filename).stem}-tr.docx'
    try:
        filename_ = f'{Path(ns.filename).stem}.docx'
        ns.filename = filename_
    except Exception as e:
        logger.error(e)
        filename_ = ns.filename
        logger.info(f'{ns.filename=}, {filename_=}')

    button_dl = mo.download(
        data=ns.docx_file,
        filename=filename_,
        label="dl docx",
    )

    file_tab = mo.vstack([
        # mo.hstack([fileobj]),
        mo.hstack(
            [fileobj, button_dx, button_lm, button_reset_pbar, button_dl,],
            align='start',
            widths=[4, 1, 1, 1, 1]
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

    # _ = ns  # react to ns? to update lmtool_tab

    lmtool_tab = mo.vstack(
        [
            slider_row_numb,
            row1,
            row2,
            row3,
        ]
    )
    return button_dl, file_tab, filename_, lmtool_tab, slider_row_numb


@app.cell
def _(mo):
    settings = mo.vstack(
        [
            mo.md("Edit User"),
            first := mo.ui.text(label="First Name"),
            last := mo.ui.text(label="Last Name"),
        ]
    )


    info = mo.vstack(
        [
            # mo.md("## Info"),
            mo.md(
                """
                * 👉 Workflow
                    - Upload a text file or a 1-3 column csv (TODO)
                    - In case of a text file, click the dxtr button to translate the text to Chinese via deeplx, lmtr button via llm, optionally download a docx file
                    - Switch to the Llmtool tab and select a paragraph using the slider. Chat with the bot. The corresponding paragraph serves as context (click ... in the up-right corner then select `Show code` for details).
                * 🐧 Join qqgroup 316287378 for updates and/or realtime chat.
                * ֎🇦🇮 If you have a  linux.do account, you may wish to check 
                this out: 
                [https://horizon.dattw.eu.org](https://horizon.dattw.eu.org) 
                (a free service courtesy of yours sincerely)"""
            ),
        ]
    )

    mo.sidebar(
        [
            # mo.md("## lmtool"),
            mo.nav_menu(
                {
                    "#/": f"{mo.icon('lucide:file')} File",
                    "#/lmtool": f"{mo.icon('lucide:bot')} Lmtool",
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
    return first, info, last, settings


@app.cell
def _(file_tab, info, lmtool_tab, mo):
    mo.routes({
        # "#/": mo.md("# Home"),
        # "#/": settings,
        "#/": file_tab,
        "#/lmtool": lmtool_tab,
        "#/info": info,
        mo.routes.CATCH_ALL: file_tab,
    })
    return


@app.cell
def _(gen_dxtext, gen_lmtext, mo, reset_pbar):
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

    button_reset_pbar = mo.ui.run_button(
        label='🚮',
        kind='warn',
        tooltip='click to remove progressbar',
        on_change=lambda _: reset_pbar()
    )
    return button_dx, button_lm, button_reset_pbar


@app.cell
def _(mo):
    import os

    os.environ.update(
        OPENAI_BASE_URL='https://litellm.dattw.eu.org/v1',
    )

    get_model, set_model = mo.state('gpt-4o-mini')
    return get_model, os, set_model


@app.cell
def _(get_model, mo, set_model):
    model_list = [
        "gpt-4o-mini", 
        "gemini-1.5-flash", "gemini-1.5-pro", 
        "gemini-2.0-flash-exp", 
        "deepseek-chat", "deepseek-v3",
    ]

    dropdown_model_sel = mo.ui.dropdown(
        options=model_list, 
        value="gpt-4o-mini", 
        label="model: ",
        on_change=set_model,
    )

    oai = mo.ai.llm.openai(
        get_model(),
        # 'gpt-4o-mini',
        system_message='You are a helpful assistant',
    )
    return dropdown_model_sel, model_list, oai


@app.cell
def _(dropdown_model_sel, mo, ns, oai, y):
    def my_model(messages, config):
        question = messages[-1].content
        y(question)

        # Search for relevant docs in a vector database, blog storage, etc.
        # docs = find_relevant_docs(question)
        # context = "\n".join(docs)
        # context = '''我叫老王。我来自北京'''
        context = f'''
            <原文:>{ns.text[ns.row_numb]}</原文:> 

            <dx译文:>{ns.dxtext[ns.row_numb]}</dx译文:>

            <lm译文:>{ns.lmtext[ns.row_numb]}</lm译文:>
        '''
        y(context)
        prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"
        # Query your own model or third-party models
        # response = query_llm(prompt, config)
        response = oai([mo.ai.llm.ChatMessage(role='user', content=prompt)], config)

        return response

    bot = mo.vstack(
        [
            dropdown_model_sel,
            mo.ui.chat(
              my_model,
              prompts=[
                "{{phrase}} 什么意思，怎么翻译",
                "列出 {{phrase}} 的 5 个中文同义词", 
                  "{{text}} 翻成中文",
                  "根据原文和dx译文给出修改后的翻译，用黑体标出新增加的文字",
                  "根据原文和lm译文给出修改后的翻译，用黑体标出新增加的文字",
                  "根据原文和dx译文及lm译文给出修改后的翻译，用黑体标出新增加的文字",
                  "根据原文和dx译文提出修改dx译文的建议",
                  "根据原文和dx译文提出修改dx译文的建议并给出修改后的翻译，用黑体标出新增加的文字",
                  "根据原文和lm译文提出修改lm译文的建议",
                  "根据原文和lm译文提出修改lm译文的建议并给出修改后的翻译，用黑体标出新增加的文字",
                  "根据原文和dx译文及lm译文提出修改建议并给出修改后的翻译，用黑体标出新增加的文字",
                "Say this is a test.",  
              ],
              show_configuration_controls=True,
              max_height=400,
            )
        ]
    )

    bot
    return bot, my_model


if __name__ == "__main__":
    app.run()
