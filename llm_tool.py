# https://docs.taipy.io/en/develop/tutorials/visuals/2_using_tables/

import asyncio
import io
import os
from docx import Document
from pathlib import Path
import pandas as pd
import numpy as np
from threading import Thread
from time import sleep, monotonic
from itertools import zip_longest

from taipy.gui import Gui, Markdown, navigate, notify, State, invoke_callback, get_state_id
import pandas as pd
from ycecream import y
from loadtext import loadtext
from loguru import logger
from langchain_openai import ChatOpenAI

from lmtr_agents import agent_tr, agent_ref, agent_imp, agent_comb, agent_comb_imp, agent_comb_imp1

from deeplx_tr import scrape_deeplx_shodan
from deeplx_tr.batch_deeplx_tr import batch_deeplx_tr
from deeplx_tr.batch_newapi_tr import batch_newapi_tr
from deeplx_tr.trtext2docx import trtext2docx
from deeplx_tr.duration_human import duration_human

from deeplx_tr.info import info_md
# from home import home_md

llm = ChatOpenAI(base_url='https://litellm.dattw.eu.org/v1', api_key="NA")

y.configure(sln=1, st=1)

# if HOST set, run on "0.0.0.0", default to 127.0.0.1
HOST = "0.0.0.0" if os.getenv("HOST") else None

food_df = pd.DataFrame({
    "Meal": ["Lunch", "Dinner", "Lunch", "Lunch", "Breakfast", "Breakfast", "Lunch", "Dinner"],
    "Category": ["Food", "Food", "Drink", "Food", "Food", "Drink", "Dessert", "Dessert"],
    "Name": ["Burger", "Pizza", "Soda", "Salad", "Pasta", "Water", "Ice Cream", "Cake"],
    "Calories": [300, 400, 150, 200, 500, 0, 400, 500],
})

path = ""
filename = ""
dlfilename = "dl-tr.docx"
n_deeplx = ""
status = ""
data_df = pd.DataFrame({
    "sn": [elm for elm in range(1)],
    "text": ["test this and that"],
    "dxtext": [""],
    "lmtext": [""],
})

def get_docx_content(docx: Document):
    """Convert docx to bytes."""
    bytesio = io.BytesIO()
    docx.save(bytesio)

    # can be save to a file:
    return bytesio.getvalue()

docx_content = get_docx_content(trtext2docx(["test this and that"]))
# Path('test1.docx').write_bytes(docx_content)

def data_df2docx(state: State):
    text = state.data_df.text.tolist()
    dxtext = state.data_df.dxtext.tolist()
    lmtext = state.data_df.lmtext.tolist()
    try:
        file_content = trtext2docx(text, dxtext, lmtext)
    except Exception as e:
        logger.error(e)
        file_content = trtext2docx([str(e)])
    return file_content

def data_df2docx_content(state: State):
    docx = data_df2docx(state)
    state.docx_content = get_docx_content(docx)

table_properties = {
    "class_name": "rows-bordered rows-similar", # optional
    # "style": table_style,
}

# can also use regular Python lists or NumPy arrays

# main_md = Markdown("<|{food_df}|table|>")
# main_md = Markdown("<|{food_df}|table|class_name=rows-bordered|>")
# main_md = Markdown("<|{food_df}|table|filter=True|>")
# main_md = Markdown("<|{food_df}|table|group_by[Category]=True|apply[Calories]=sum|>")
# main_md = Markdown("<|{food_df}|table|group_by[Category]=True|apply[Calories]=sum|filter=True|>")

# https://docs.taipy.io/en/develop/tutorials/visuals/6_css_style_kit/

def txt2list(state):
    try:
        path = state.path
        y(path)
    except Exception as exc:
        logger.error(exc)
        return None
    if Path(path).exists():
        try:
            texts = loadtext(path)
        except Exception as exc:
            logger.error(exc)
            return None
        n_paras = len(texts)
        y(n_paras)

        _ = len(str(n_paras - 1))
        sn_list = [f"{elm:0{_}}" for elm in range(n_paras)]
        y(sn_list[:5])

        state.data_df = pd.DataFrame(
            np.array([*zip_longest(
                sn_list,
                texts,
                [],
                [],
                fillvalue="",
            )]),
            columns=["sn", "text", "dxtext", "lmtext"],
        )

    state.filename = Path(state.path).name
    state.dlfilename = f"{Path(state.filename).stem}-tr.docx"

    y(state.filename, state.dlfilename)

    y(state.data_df[:5])


# sort of a global
state_id = []

def on_init(state: State):
    y("enter on_init")
    state_id.append(get_state_id(state))
    y(state_id)
    state.n_deeplx = "diggin..."
    y("exit on_init")


def set_n_deeplx_handler(gui: Gui):
    y("enter set_n_deeplx_handler")
    while not state_id:  # wait for on_init to finish
        sleep(0.5)
    invoke_callback(gui, state_id[0], set_n_deeplx, ())
    y("exit set_n_deeplx_handler")


def set_n_deeplx(state: State):
    y("enter set_n_deeplx")
    while True:
        state.n_deeplx = "diggin..."
        try:
            total = asyncio.run(scrape_deeplx_shodan.main())
        except Exception as e:
            logger.error(e)
            total = "errors: " + str(e)[:5] + "..."
        state.n_deeplx = total
        sleep(20 * 60)  # update deeplx-urls in diskcacache every 20 minutes
    y("exit set_n_deeplx")


def deepl_tr_action(state: State):
    """
    Translate, deeplx-tr|button, data_df["text"] -> data_df["dxtext"].

    using batch_deeplx_tr()
    """
    y("enter deepl_tr_action")
    then = monotonic()
    state.status = "diggin deepx_tr..."
    try:
        texts = state.data_df.text.to_list()
        y(texts)
        trtext_2 = asyncio.run(batch_deeplx_tr(texts))
        y(trtext_2)
    except Exception as e:
        logger.error(e)
        trtext_2 = []
        state.status = str(e)
        sleep(2)
    if trtext_2:
        len_ = len(state.data_df)
        dict_ = dict(trtext_2)
        dxtex = [dict_.get(elm, "") for elm in range(len_)]

        state.data_df.dxtext = dxtex
        y(state.data_df[:3])

        state.refresh("data_df")

        data_df2docx_content(state)

    state.status = duration_human(monotonic() - then)
    y("done deepl_tr_action")


def llm_tr_action(state: State):
    """
    Translate, deeplx-tr|button, data_df["text"] -> data_df["dxtext"].

    using batch_deeplx_tr()
    """
    y("enter llm_tr_action")
    then = monotonic()
    state.status = "diggin llm_tr..."
    try:
        texts = state.data_df.text.to_list()
        y(texts)
        trtext_2 = asyncio.run(batch_newapi_tr(texts))
        y(trtext_2)
    except Exception as e:
        logger.error(e)
        trtext_2 = []
        state.status = str(e)
        sleep(2)
    if trtext_2:
        len_ = len(state.data_df)
        dict_ = dict(trtext_2)
        lmtex = [dict_.get(elm, "") for elm in range(len_)]

        state.data_df.lmtext = lmtex
        y(state.data_df[:3])

        state.refresh("data_df")

        data_df2docx_content(state)

    state.status = duration_human(monotonic() - then)
    y("done llm_tr_action")


def save_docx_action(state: State):
    """Save docx, save-docx|button."""
    y("enter save_docx_action")

# {: .color-primary}
# **LLM** **Tool**
# {: .text-center}

show_pane = False
show_dialog = True
buffer = " "
reflection = " "
response = " "
row_no = 0

# def dialog_action(state, id, payload):
def toggle_table_dialog1(state):
    with state as st:
        ...
        # depending on payload["args"][0]: -1 for close icon, 0 for Validate, 1 for Cancel
        ...
        st.show_dialog = not st.show_dialog

# dialog pop-up
def toggle_table_dialog(state):
    state.show_table_dialog = not state.show_table_dialog


show_table_dialog = False

# {: .align-columns-top .align-columns-right}

main_md = Markdown("""
<|card card-bg|
<|layout|columns=1 1 1 1|

<|{path}|file_selector|label=Pick .txt File|extensions=.txt|on_action=txt2list|drop_message=Drop here|>

<|deeplx-tr|class_name=error|button|on_action=deepl_tr_action|hover_text=Click to translte text column via deeplx, it will be quite fast if n-deeplx-urls is large relative to number of parahraphs (= 1 + maximum of sn), for example, 50 paragraphs of average lengths will take just a few seconds.|>

<|llm-tr|button|on_action=llm_tr_action|hover_text=Click to translte text column via llms, it will be slow. For example, 100 paragraphs of average lengths will likely take around a few minutes.|>

<|{docx_content}|file_download|label=Dl tr.docx|name=dl-tr.docx|>

<|{filename}|>

n-deeplx-urls: <|{n_deeplx}|>

status: <|{status}|>

|>
|>

<|{data_df}|table|class_name=rows-bordered|properties=table_properties|>

<|part|partial={llm_suggest_partial}|>

""")

# hover_over=docx format with colored and higglighted text for easy one-click hide op.|
# dl-tr.docx {dlfilename}
# <|save-docx|button|on_action=save_docx_action|hover_text=Currently, only docx format is available|>

_ = """
<|{show_pane}|pane|persistent|width=100px|anchor=right|
Pane content
|>
<|OpenPanel|button|on_action={lambda s: s.assign("show_pane", True)}|>

<|Dialog|button|on_action=toggle_table_dialog|>
<|{show_table_dialog}|dialog|on_action=toggle_table_dialog|width=90vw|labels={["Cancel"]}|
    Dialog
|>
# """

# <|{show_dialog}|dialog|title=Dialog Title|on_action=toggle_table_dialog|labels=OK;Cancel|

_ = """
pages = {
    "/": "<|navbar|>",
    # "home": home_md,
    "main": main_md,
    "info": info_md,
}
# """

pages = {
    "/": "<|menu|lov={page_names}|on_action=menu_action|>",
    "main": main_md,
    "info": info_md,
}
page_names = [page for page in pages.keys() if page != "/"]

def menu_action(state, action, payload):
    y(action)
    y(payload)

    page = payload["args"][0]
    navigate(state, page)
    # navigate(state, action)

def send_text_action(state, row=None):
    """Send data_df.text[row] to buffer."""
    y(state.buffer)
    y(row)

    if not row or row == 0:
        try:
            row = int(state.row_no)
        except Exception as exc:
            logger.error(exc)
            row = 0

    try:
        y(state.data_df.text[row])
        state.buffer = state.data_df.text[row]
    except Exception as e:
        logger.warning(f"{e=}: {row=}")

    # _ = state.data_df.copy()
    # _.loc[0, "text"] = row
    # state.data_df = _

def send_dxtext_action(state, row=None):
    """Send data_df.dxtext[row] to buffer."""
    y(state.buffer)
    y(row)
    if not row or row == 0:
        try:
            row = int(state.row_no)
            y(state.data_df.dxtext[row])
            state.buffer = state.data_df.dxtext[row]
        except Exception as e:
            logger.warning(f"{e}: {row=}")


def send_lmtext_action(state, row=None):
    """Send data_df.lmtext[row] to buffer."""
    y(state.buffer)
    y(row)
    if not row or row == 0:
        try:
            row = int(state.row_no)
            y(state.data_df.lmtext[row])
            state.buffer = state.data_df.lmtext[row]
        except Exception as e:
            logger.warning(f"{e}: {row=}")


def send_text_action1(state, row=1):
    y(row)


send_text_action2 = lambda x: send_text_action(x, row=2)
globals()["send_text_action3"] = lambda x: send_text_action(x, row=3)


def chat_action(state):
    """Send chat response to Output."""
    y("enter chat_action -- coming soon, stay tuned")
    if not state.buffer.strip():
        state.response = "Buffer empty -- first click one of TEXT, TRTEXT LMTEXT. Or type something in the Buffer box."
        return
    state.response = " llm diggin..."
    try:
        resp = llm.invoke(state.buffer)
        # response = f"{resp.content} ({resp.usage_metadata})"
        response = resp.content
    except Exception as exc:
        logger.error(exc)
        response = str(exc)
    state.response = " chat_action -- coming soon, stay tuned"
    state.response = response

def reflect_action(state):
    """Send (imporvement) advice response to Output."""
    y("enter reflect_action -- coming soon, stay tuned")
    if not state.buffer.strip():
        state.reflection = "Buffer empty, nothing to relefct on -- first click one of TRTEXT LMTEXT."
        return
    state.reflection = " reflect_action diggin..."

    if not state.row_no or state.row_no == 0:
        try:
            row = int(state.row_no)
        except Exception as e:
            logger.warning(f"{e}: {row=}")

    y(state.data_df.text[row])
    y(state.buffer)

    try:
        reflection = agent_ref(state.data_df.text[row], state.buffer)
    except Exception as exc:
        reflection = str(exc)

    state.reflection = "reflect_action -- coming soon, stay tuned"
    state.reflection = reflection

def improve_action(state):
    """Send (imporvement) advice response to Output."""
    y("enter improve_action -- coming soon, stay tuned")
    if not state.buffer.strip():
        state.response = "Buffer empty -- first click one of TEXT, TRTEXT LMTEXT."
        return
    state.response = "improve_action -- coming soon, stay tuned"

def combine_action(state):
    """Send (imporvement) advice response to Output."""
    y("enter combine_action -- coming soon, stay tuned")
    if not state.buffer.strip():
        state.response = "Buffer empty -- first click one of TEXT, TRTEXT LMTEXT."
        return
    state.response = "combine_action -- coming soon, stay tuned"

def combimp_action(state):
    """Send (imporvement) advice response to Output."""
    y("enter combimp_action -- coming soon, stay tuned")
    if not state.buffer.strip():
        state.response = "Buffer empty -- first click one of TEXT, TRTEXT LMTEXT."
        return
    state.response = "combimp_action -- coming soon, stay tuned"


_ = """
Gui(pages=pages).run(
    title="llm translate tool",
    use_reloader=True,
    debug=True,
    dark_mode=False,
    host=HOST,
)
# """

partial_md = """
<|

<|layout|gap=0px|columns=80px 80px 120px 120px 1fr|

<|{row_no}|input|label=row-n|hover_text=specify n-th row, click text or dxtext or lmtext to send text to Buffer for processsing|>

<|Text|button|on_action=send_text_action|>

<|dxtext|button|on_action=send_dxtext_action|>

<|lmtext|button|on_action=send_lmtext_action|>

<|
<|Chat|button|class_name=primary|on_action=chat_action|>
<|Reflect|button|class_name=primary|on_action=reflect_action|>
<|Improve|button|class_name=primary|on_action=improve_action|>
<|Combine|button|class_name=primary|on_action=combine_action|>
<|Combimp|button|class_name=primary|on_action=combimp_action|>
|>

|>

<|layout|columns=1fr 1.2fr 1fr|

<|{buffer}|input|multiline=True|label=Buffer|>

<|{reflection}|input|multiline=True|label=Reflection|>

<|{response}|input|multiline=True|label=LLM Response|>

|>

|>
"""

gui = Gui(pages=pages)

llm_suggest_partial = gui.add_partial(partial_md)
# <layout|columns=1 1 1 1|
# |>

if __name__ == "__main__":
    def signal_handler(signal, frame):
        print(f"signal: {signal}")
        print(f"frame: {frame}")
        print("You pressed Ctrl+C, goodbye!")
        sys.exit(130)  # raise SystemExit(130)
    # or use
    # signal.signal(signal.SIGINT, signal_handler)

    Thread(target=set_n_deeplx_handler, args=(gui,)).start()

    gui.run(
        title="llm translate tool",
        use_reloader=True,
        debug=True,
        dark_mode=False,
        host=HOST,
    )
