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

# from ycecream import y

# y.configure(sln=True)

st.set_page_config(
    page_title="Hello1111",
    page_icon="👋",
)
if "ns" not in st.session_state:
    st.session_state["ns"] = Box()

sstate = st.session_state

if sstate.ns.get("dataframe") is None:
    sstate.ns.dataframe = pd.DataFrame([[""] * 3], columns=["text", "dxtext", "lmtext"])

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
    if st.toggle("toggle", value=True):
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
            if "texts" not in st.session_state:
                st.session_state["texts"] = texts
            sstate.ns.texts = texts

            dataframe = pd.DataFrame(zip_longest(texts, [], [], fillvalue=""), columns=["text", "dxtext", "lmtext"])
            sstate.ns.dataframe = dataframe

            sstate.ns.filename = uploaded_file.name
            # st.write("Filename: ", sstate.ns.filename)
if sstate.ns.get("filename") is not None:
    fn_placeholder.text(f"file: {sstate.ns.filename}")

with row0[1]:
    if st.button("dxtr", type="primary", key="dxtr"):
        # st.write(randrange(10))
        # st.session_state.dataframe = None
        # sstate.ns.dataframe = None
        err = "there is a problem with deeplx, notify the dev of this tool if possible"
        try:
            trtext_2 = asyncio.run(batch_deeplx_tr(sstate.ns.texts))
        except Exception as e:
            logger.error(e)
            err = str(e)
            trtext_2 = []
            placeholder.text(f"{e=}")
        if trtext_2:
            len_ = len(sstate.texts)
            dict_ = dict(trtext_2)
            dxtext = [dict_.get(elm, "") for elm in range(len_)]
        else:
            dxtext = [err]
        dataframe = pd.DataFrame(zip_longest(sstate.ns.texts, dxtext, [], fillvalue=""), columns=["text", "dxtext", "lmtext"])
        sstate.ns.dxtext = dxtext
        sstate.ns.dataframe = dataframe

with row0[2]:
    if st.button("lmtr", type="primary", key="lmtr"):
        # st.write(randrange(10))
        # st.session_state.dataframe = None
        placeholder.empty()
        placeholder.text("diggin...")
        sleep(5)
        placeholder.text("done")
        err = "there is a problem with lm translate, notify the dev of this tool if possible"
        try:
            trtext_2 = asyncio.run(batch_newapi_tr(sstate.ns.texts))
        except Exception as e:
            logger.error(e)
            err = str(e)
            trtext_2 = []
            placeholder.text(f"{e=}")
        if trtext_2:
            len_ = len(sstate.texts)
            dict_ = dict(trtext_2)
            lmtext = [dict_.get(elm, "") for elm in range(len_)]
        else:
            lmtext = [err]
        dataframe = pd.DataFrame(zip_longest(sstate.ns.texts, dxtext, lmtext, fillvalue=""), columns=["text", "dxtext", "lmtext"])
        sstate.ns.lmtext = lmtext
        sstate.ns.dataframe = dataframe
        
with row0[3]:
    if st.button("dl-file", type="primary", key="dl-file"):
        st.write(randrange(10))

logger.trace(f"{sstate.ns.dataframe=}")

st.data_editor(sstate.ns.dataframe, use_container_width=True)
# st.dataframe(sstate.ns.dataframe, use_container_width=True)

if sstate.ns.get("texts") is not None:
    logger.trace(f"{sstate.ns.texts=}")
if sstate.ns.get("filename") is not None:
    logger.trace(f"{sstate.ns.filename=}")

# placeholder.text("diggin...")
# sleep(5)
# placeholder.text("Done")

# st.button("Regenerate")
