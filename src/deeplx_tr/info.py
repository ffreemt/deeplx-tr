from taipy.gui import Markdown, navigate

text = "Welcome to deeplx/llm tools for translation!"

def go_home(state):
    navigate(state, "main")

# <|navbar|>
# info_md = """
info_md = Markdown("""

# Info

<|{text}|>

* Join qq-group 桃花元@316287378 for updates and real-time chat.

* Get your own free APIKEY from [https://newapi.dattw.eu.org](https://newapi.dattw.eu.org).

<|To Main|button|on_action=go_home|>

<|toggle|theme|>

""")
# """