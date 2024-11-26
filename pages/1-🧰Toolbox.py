import streamlit as st
import time
import numpy as np
from box import Box
from langchain_openai import ChatOpenAI
from lmtr_agents import (
    agent_imp,
    # agent_tr,
    agent_ref,
)
from loguru import logger
from deeplx_tr.color_diff import color_diff, plussign_diff

llm = ChatOpenAI(
    base_url="https://litellm.dattw.eu.org/v1",
    api_key="NA",
    # model="gemini-pro",
    # model="deepseek-chat",
    model="gemini-1.5-pro",
)
# llm.invoke("Enbrace什么意思")
st.set_page_config(page_title="Translate Agents", page_icon="🧰")

sstate = st.session_state
if "ns" not in st.session_state:
    st.session_state["ns"] = Box()

if sstate.ns.get("text") is None:
    sstate.ns.text = [""]
if sstate.ns.get("dxtext") is None:
    sstate.ns.dxtext = [""]
if sstate.ns.get("lmtext") is None:
    sstate.ns.lmtext = [""]
if sstate.ns.get("filename") is None:
    sstate.ns.filename = "temp.txt"
if sstate.ns.get("flag") is None:
    sstate.ns.flag = True
if sstate.ns.get("sn") is None:
    sstate.ns.sn = 0
if sstate.ns.get("reflection") is None:
    sstate.ns.reflection = ""

# st.markdown("# Plotting Demo")
# st.markdown("# LLM-Toolbox")
# st.sidebar.header("Plotting Demo")

placeholder = st.sidebar.empty()

# diggin = st.sidebar.info("diggin...")

sn = 0
len_ = len(sstate.ns.text)
if len_ < 2:
    range_ = [0, 1]
else:
    range_ = [elm for elm  in range(len(sstate.ns.text))]

if sstate.ns.filename == "temp.txt":
    placeholder.write("no file loaded, load a file first in 🌐Translate")
else:
    # placeholder.empty()
    if sstate.ns.flag:
        placeholder.write("Move the red dot to picke a para to work with")
        sstate.ns.flag = False  # just show once

row_text = st.columns([1, 10])
slider = st.empty()

with row_text[1]:
    para_text = st.empty()

# _ = """
sn0 = slider.select_slider(
    "Select a para",
    options=range_,
    # value=sstate.ns.sn,
)
# """

css = """
<style>
    .stNumberInput label {
        display: none;
    }
    .stSlider label {
        display: none;
    }
</style>
"""

# show text dxtext lmtext
with row_text[0]:
    sn = st.number_input("sn", min_value=0, max_value=len_ - 1, value=sn0, key="keysn")
if 0 <= sn < len_:
    # para_text.write(f"{sn}, {sstate.ns.text[sn]}")
    para_text.write(
        sstate.ns.text[sn] +
        "\n\n" +
        sstate.ns.dxtext[sn] +
        "\n\n" +
        sstate.ns.lmtext[sn]
    )
    sstate.ns.sn = sn

st.markdown(css, unsafe_allow_html=True)

_ = "Ask a question (e.g., x 什么意思) or make a request (e.g. 翻成中文)"
if prompt := st.chat_input(_):
    st.chat_message("user").markdown(prompt)
    response = f"Echo: {prompt} 说中文"

    _ = f"{sstate.ns.text[sn]}\n{prompt}, 说中文"
    placeholder.text("diggin llm.invoke...")
    try:
        resp = llm.invoke(_)
        # response = f"{resp.content} ({resp.usage_metadata})"
        response = resp.content
    except Exception as exc:
        logger.error(exc)
        response = str(exc)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    placeholder.text("done llm.invoke...")

reflect_text = st.empty()
improve_text = st.empty()

row_agents = st.columns(4)
with row_agents[0]:
    if st.button("reflectdx", type="primary", key="reflectdx"):
        placeholder.empty()
        placeholder.text("diggin reflectdx...")
        try:
            sn = sstate.ns.sn
            reflection = agent_ref(sstate.ns.text[sn], sstate.ns.dxtext[sn])
            placeholder.text("done reflectdx")
        except Exception as exc:
            reflection = f"{exc=}, net hiccup? try again"
            # state.status = "uh oh... try again"
            placeholder.text("uh oh... try again")
        sstate.ns.reflection = reflection
        reflect_text.markdown(reflection)

with row_agents[1]:
    if st.button("improvedx", type="primary", key="improvedx"):
        placeholder.empty()
        placeholder.text("diggin imporvedx...")

        try:
            sn = sstate.ns.sn
            text = sstate.ns.text[sn]
            dxtext = sstate.ns.dxtext[sn]
            reflection = sstate.ns.reflection
            response = agent_imp(text, dxtext, reflection)
            # highlight
            _ = """
            improve = (
                plussign_diff(dxtext, response)
                + "\n === TODO \n<br/>"
                + color_diff(dxtext, response)
            )
            # """
            improve = color_diff(dxtext, response)
            # improve = response

            placeholder.text("done imporvedx")
        except Exception as exc:
            improve = f"{exc=}, net hiccup? try again"
            placeholder.text("uh oh... try again")

        improve_text.markdown(improve, unsafe_allow_html=True)

with row_agents[2]:
    if st.button("reflectlm", type="primary", key="reflectlm"):
        placeholder.empty()
        placeholder.text("diggin reflectlm...")
        try:
            sn = sstate.ns.sn
            reflection = agent_ref(sstate.ns.text[sn], sstate.ns.lmtext[sn])
            placeholder.text("done reflectlm")
        except Exception as exc:
            reflection = f"{exc=}, net hiccup? try again"
            # state.status = "uh oh... try again"
            placeholder.text("uh oh... try again")
        sstate.ns.reflection = reflection
        reflect_text.markdown(reflection)

with row_agents[3]:
    if st.button("improvelm", type="primary", key="improvelm"):
        placeholder.empty()
        placeholder.text("diggin imporvelm...")

        try:
            sn = sstate.ns.sn
            text = sstate.ns.text[sn]
            lmtext = sstate.ns.lmtext[sn]
            reflection = sstate.ns.reflection
            response = agent_imp(text, lmtext, reflection)
            # highlight
            _ = """
            improve = (
                plussign_diff(lmtext, response)
                + "\n === TODO \n<br/>"
                + color_diff(lmtext, response)
            )
            # """
            improve = color_diff(lmtext, response)
            # improve = response

            placeholder.text("done imporvelm")
        except Exception as exc:
            improve = f"{exc=}, net hiccup? try again"
            placeholder.text("uh oh... try again")

        improve_text.markdown(improve, unsafe_allow_html=True)
