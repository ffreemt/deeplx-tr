"""Define Translate tab."""

import asyncio
import io
from io import StringIO
from itertools import zip_longest
import json
from pathlib import Path
from time import monotonic

import pandas as pd
import streamlit as st
from box import Box
from deeplx_tr.batch_deeplx_tr import batch_deeplx_tr
from deeplx_tr.batch_newapi_tr import batch_newapi_tr
from deeplx_tr.trtext2docx import trtext2docx
from loguru import logger

logger.trace(" session start ------------------------ ")
logger.info(" session start ------------------------ ")

st.set_page_config(
    page_title="Translate",
    page_icon="🌐",
)

# if "ns" not in st.session_state:
#   st.session_state["ns"] = Box()

sstate = st.session_state

# load streamlit_sstate_cache.json if present
# cache is on the server, only useful when the app restarts
cache_file = "streamlit_sstate_cache.json"
if "ns" not in sstate:
    try:
        sstate["ns"] = Box(json.loads(Path(cache_file).read_text("utf8")))

        # also update sstate.dataframe and view
        sstate["dataframe"] = pd.DataFrame(sstate.ns.json)

        logger.info(f"{sstate.dataframe=}")
        logger.info(f" {cache_file} oaded to st.session_state['ns'] ")
    except Exception:
        sstate["ns"] = Box()


# to save back:
@st.cache_data
def save_sstate_ns(obj):
    # Path(cache_file).write_text(json.dumps(sstate["ns"]))
    Path(cache_file).write_text(json.dumps(obj))


if sstate.ns.get("text") is None:
    sstate.ns.text = [""]
if sstate.ns.get("dxtext") is None:
    sstate.ns.dxtext = [""]
if sstate.ns.get("lmtext") is None:
    sstate.ns.lmtext = [""]

if sstate.ns.get("filename") is None:
    sstate.ns.filename = "temp.txt"

# dataframe cannot be serialized, hence cannot be saved
# convert to json first
if "dataframe" not in sstate:
    sstate["dataframe"] = pd.DataFrame(
        [[""] * 3], columns=["text", "dxtext", "lmtext"]
    )

if sstate.ns.get("json") is None:
    sstate.ns.json = sstate.dataframe.to_json()

# toolbox_pos
if sstate.ns.get("toolbox_pos") is None:
    sstate.ns.toolbox_pos = 0

# y(sstate.ns)  # this does not work
logger.trace(f"{sstate.ns.keys()}")

fn_placeholder = st.sidebar.empty()
placeholder = st.sidebar.empty()

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
        sstate.ns.text = texts

        dataframe = pd.DataFrame(
            zip_longest(sstate.ns.text, [], [], fillvalue=""),
            columns=["text", "dxtext", "lmtext"],
        )

        # cant only save list
        sstate.ns.json = dataframe.to_json()

        sstate.ns.filename = uploaded_file.name
        # st.write("Filename: ", sstate.ns.filename)
        try:
            save_sstate_ns(sstate["ns"])
        except Exception as e:
            logger.error(e)

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
        dataframe = pd.DataFrame(
            zip_longest(
                sstate.ns.text, sstate.ns.dxtext, sstate.ns.lmtext, fillvalue=""
            ),
            columns=["text", "dxtext", "lmtext"],
        )
        sstate.dataframe = dataframe

with row0[2]:
    if st.button("lmtr", type="primary", key="lmtr"):
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
        dataframe = pd.DataFrame(
            zip_longest(
                sstate.ns.text, sstate.ns.dxtext, sstate.ns.lmtext, fillvalue=""
            ),
            columns=["text", "dxtext", "lmtext"],
        )
        sstate.dataframe = dataframe


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



csvdata = convert_df(sstate.dataframe)
docxdata = convert_df2docx(sstate.dataframe)
with row0[3]:
    _ = """
    if st.button("dl-file", type="primary", key="dl-file"):
        st.write(randrange(10))
        logger.debug(f">>> {sstate.dataframe=}")
    else:
        logger.debug(">>> ")
    """
    # .docx     application/vnd.openxmlformats-officedocument.wordprocessingml.document
    # "application/octet-stream"
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


dataframe = pd.DataFrame(
    zip_longest(sstate.ns.text, sstate.ns.dxtext, sstate.ns.lmtext, fillvalue=""),
    columns=["text", "dxtext", "lmtext"],
)
sstate.dataframe = dataframe

logger.trace(f"{sstate.dataframe=}")
logger.info(f"{sstate.dataframe=}")

logger.info(" st.data_editor ...")
# row1 view
# st.data_editor(sstate.dataframe)
st.data_editor(
    sstate.dataframe,
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
    },
)

# st.dataframe(sstate.dataframe, use_container_width=True)

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

logger.info(" ######################## session end ")
logger.trace(" ######################## session end ")
