import asyncio
import io
import os
import signal
import sys
from itertools import zip_longest
from pathlib import Path
from threading import Thread
from time import monotonic, sleep

import numpy as np
import pandas as pd

import streamlit as st
import pandas as pd
import numpy as np

from io import StringIO, BytesIO
from random import randrange
from itertools import zip_longest
from box import Box
from loguru import logger

from deeplx_tr import scrape_deeplx_shodan
from deeplx_tr.batch_deeplx_tr import batch_deeplx_tr
from deeplx_tr.batch_newapi_tr import batch_newapi_tr
from deeplx_tr.color_diff import color_diff, plussign_diff
from deeplx_tr.duration_human import duration_human
from deeplx_tr.info import info_md
from deeplx_tr.trtext2docx import trtext2docx

logger.trace(" ------------------------ ")

st.set_page_config(
    page_title="Translate",
    page_icon="🌐",
)
if "ns" not in st.session_state:
    st.session_state["ns"] = Box()

sstate = st.session_state
if sstate.ns.get("text") is None:
    sstate.ns.text = [""]
if sstate.ns.get("dxtext") is None:
    sstate.ns.dxtext = [""]
if sstate.ns.get("lmtext") is None:
    sstate.ns.lmtext = [""]

if sstate.ns.get("dataframe") is None:
    sstate.ns.dataframe = pd.DataFrame([[""] * 3], columns=["text", "dxtext", "lmtext"])

if sstate.ns.get("filename") is None:
    sstate.ns.filename = "temp.txt"

# y(sstate.ns)  # this does not work
logger.trace(f"{sstate.ns.keys()}")

fn_placeholder = st.sidebar.empty()
placeholder = st.sidebar.empty()

if "dataframe" not in sstate:
    st.session_state["dataframe"] = pd.DataFrame([[""] * 3], columns=["text", "dxtext", "lmtext"])

# @st.fragment()
# def toggle_uploaded_file():
row0 = st.columns(4)
with row0[0]:
    # if st.toggle("toggle", value=True):
    uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False)
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        # st.write(bytes_data)

        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # st.write(stringio)

        # To read file as string:
        string_data = stringio.read()
        # Can be used wherever a "file-like" object is accepted:
        # dataframe = pd.read_csv(uploaded_file)
        # dataframe = loadtext(string_data)

        texts = [elm for elm in string_data.splitlines() if elm.strip()]
        if "text" not in st.session_state:
            st.session_state["text"] = texts
        sstate.ns.text = texts

        dataframe = pd.DataFrame(zip_longest(sstate.ns.text, [], [], fillvalue=""), columns=["text", "dxtext", "lmtext"])
        sstate.ns.dataframe = dataframe

        sstate.ns.filename = uploaded_file.name
        # st.write("Filename: ", sstate.ns.filename)

hide_label = """
<style>
    .st-emotion-cache-1fttcpj {
        display: none;
    }
</style>
"""
st.markdown(hide_label, unsafe_allow_html=True)

if sstate.ns.get("filename") is not None:
    if "temp.txt" not in sstate.ns.filename:
        fn_placeholder.text(f"file: {sstate.ns.filename}")

with row0[1]:
    if st.button("dxtr", type="primary", key="dxtr"):
        # st.write(randrange(10))
        # st.session_state.dataframe = None
        # sstate.ns.dataframe = None
        placeholder.text("diggin dxtr...")
        then = monotonic()
        err = "there is a problem with deeplx, notify the dev of this tool if possible"
        try:
            trtext_2 = asyncio.run(batch_deeplx_tr(sstate.ns.text))
        except Exception as e:
            logger.error(e)
            err = str(e)
            trtext_2 = []
            placeholder.text(f"{e=}")
        if trtext_2:
            len_ = len(sstate.ns.text)
            dict_ = dict(trtext_2)
            dxtext = [dict_.get(elm, "") for elm in range(len_)]
        else:
            dxtext = [err]
        sstate.ns.dxtext = dxtext
        placeholder.text(f"done dxtr {monotonic() - then:.2f} ")
        dataframe = pd.DataFrame(zip_longest(sstate.ns.text, sstate.ns.dxtext, sstate.ns.lmtext, fillvalue=""), columns=["text", "dxtext", "lmtext"])
        sstate.ns.dataframe = dataframe

with row0[2]:
    if st.button("lmtr", type="primary", key="lmtr"):
        # st.write(randrange(10))
        # st.session_state.dataframe = None
        placeholder.empty()
        placeholder.text("diggin lmtr...")
        then = monotonic()
        err = "there is a problem with lm translate, notify the dev of this tool if possible"
        try:
            trtext_2 = asyncio.run(batch_newapi_tr(sstate.ns.text))
        except Exception as e:
            logger.error(e)
            err = str(e)
            trtext_2 = []
            placeholder.text(f"{e=}")
        if trtext_2:
            len_ = len(sstate.ns.text)
            dict_ = dict(trtext_2)
            lmtext = [dict_.get(elm, "") for elm in range(len_)]
        else:
            lmtext = [err]
        sstate.ns.lmtext = lmtext

        placeholder.text(f"done {monotonic() - then:.2f} ")
        dataframe = pd.DataFrame(zip_longest(sstate.ns.text, sstate.ns.dxtext, sstate.ns.lmtext, fillvalue=""), columns=["text", "dxtext", "lmtext"])
        sstate.ns.dataframe = dataframe

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

@st.cache_data
def convert_df2docx(df):
    text = df.text.tolist()
    dxtext = df.dxtext.tolist()
    lmtext = df.lmtext.tolist()
    try:
        file_content = trtext2docx(text, dxtext, lmtext)
    except Exception as e:
        logger.error(e)
        file_content = trtext2docx([str(e)])
    # return file_content

    # https://discuss.streamlit.io/t/downloading-string-as-docx-format-with-st-download-button/75075
    bio = io.BytesIO()
    file_content.save(bio)
    return bio.getvalue()


csvdata = convert_df(sstate.ns.dataframe)

with row0[3]:
    _ = """
    if st.button("dl-file", type="primary", key="dl-file"):
        st.write(randrange(10))
        logger.debug(f">>> {sstate.ns.dataframe=}")
    else:
        logger.debug(">>> ")
    """
    # .docx     application/vnd.openxmlformats-officedocument.wordprocessingml.document
    # "application/octet-stream"
    docxdata = convert_df2docx(sstate.ns.dataframe)
    st.download_button(
        label="Download docx",
        data=docxdata,
        file_name=f"{Path(sstate.ns.filename).stem}-tr.docx",
        mime="docx",
        # type="primary",
    )
    st.download_button(
        label="Download csv",
        data=csvdata,
        file_name=f"{Path(sstate.ns.filename).stem}-tr.txt",
        mime="text/csv",
        # type="primary",
    )


dataframe = pd.DataFrame(zip_longest(sstate.ns.text, sstate.ns.dxtext, sstate.ns.lmtext, fillvalue=""), columns=["text", "dxtext", "lmtext"])
sstate.ns.dataframe = dataframe

logger.trace(f"{sstate.ns.dataframe=}")

# st.data_editor(sstate.ns.dataframe)
st.data_editor(
    sstate.ns.dataframe,
    use_container_width=True,
    column_config={
        "text": st.column_config.TextColumn(
            "source text",
            help="soure text",
            # default="st.",
            # max_chars=50,
            # validate=r"^st\.[a-z_]+$",
            disabled=False,
            width="small",
        ),
        "dxtext": st.column_config.TextColumn(
            "deeplxtr text",
            help="translated text using deeplx (target language: currently only Simplified Chinese)",
            # default="st.",
            # max_chars=50,
            # validate=r"^st\.[a-z_]+$",
            disabled=False,
            width="small",
        ),
        "lmtext": st.column_config.TextColumn(
            "llmtr text",
            help="translated text using llm (target language: currently only Simplified Chinese)",
            # default="st.",
            # max_chars=50,
            # validate=r"^st\.[a-z_]+$",
            disabled=False,
            width="small",
        ),
    }
)

# st.dataframe(sstate.ns.dataframe, use_container_width=True)

if sstate.ns.get("text") is not None:
    logger.trace(f"{sstate.ns.text=}")
if sstate.ns.get("dxtext") is not None:
    logger.trace(f"{sstate.ns.dxtext=}")
if sstate.ns.get("lmtext") is not None:
    logger.trace(f"{sstate.ns.lmtext=}")
if sstate.ns.get("filename") is not None:
    logger.trace(f"{sstate.ns.filename=}")
else:
    sstate.ns.filename = "temp.txt"
    logger.trace(f"{sstate.ns.filename=}")

# placeholder.text("diggin...")
# sleep(5)
# placeholder.text("Done")

# st.button("Regenerate")

logger.trace(" ######################## ")
