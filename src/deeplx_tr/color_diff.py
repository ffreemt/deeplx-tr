"""
Color (default red) the difference in text2 when compaerd with text1.

gpt prompt: how to use python to highight (use only red color font) the difference in text2 compared to text1 in markdown?

import difflib

text1 = '''所谓的 "数字双胞胎 "是复杂系统的动态虚拟复制品。组织机构通常将其用于情景规划，因为它们将现实世界的元素与模拟和持续的数据流融合在一起，有助于评估不同决策的后果。例如，当 BMO 在 2023 年收购了 503 家西部银行分行时，它使用 Matterport 的捕捉服务在三个月内创建了所有分行地点的三维数字双胞胎。'''
text2 = '''所谓的“数字双生体”是复杂系统的动态虚拟副本。组织通常将其用于情境规划，因为它们将现实世界的元素与模拟及持续的数据流相结合，有助于评估不同决策的影响。例如，当 BMO 于 2023 年收购了 503 家西部银行的分行时，它使用 Matterport 的捕获服务在三个月内为所有分行地点创建了三维数字双生体。'''

how to use python to highight (use only inline style and red color) the difference in text2 compared to text1 in markdown?

using python and inline style red color markdown, how to highlight the difference in text2 when compared to text1, only the difference must be lighlighted

using python and inline style red color markdown, how to only highlight the difference in text2 when compared to text1

# Find differences
differ = difflib.Differ()
diff = list(differ.compare(s1.split(), s2.split()))

# Highlight differences in s2
highlighted_s2 = []
for word in diff:
    if word.startswith('+ '):  # Added in s2
        highlighted_s2.append(f'<span style="color:red">{word[2:]}</span>')
    elif word.startswith('  '):  # No change
        highlighted_s2.append(word[2:])
    # Words starting with '- ' (removed from s1) are ignored

highlighted_s2_str = ' '.join(highlighted_s2)
print(highlighted_s2_str)

"""

from difflib import ndiff


def color_diff(text1: str, text2: str, color: str = "red"):
    """Color (default red) the difference in text2 when compaerd with text1."""
    diff = list(ndiff(text1, text2))

    output = ""
    for part in diff:
        if part.startswith(
            "- "
        ):  # text present in text1 but not in text2 (removed in text2)
            continue
        elif part.startswith(
            "+ "
        ):  # text present in text2 but not in text1 (added in text2)
            output += f'<span style="color:{color}">' + part[2:] + "</span>"
        else:
            output += part[2:]

    return output


def plussign_diff(text1: str, text2: str):
    """Put a plus sign before the difference in text2 when compaerd with text1."""
    diff = list(ndiff(text1, text2))

    output = ""
    for part in diff:
        if part.startswith(
            "- "
        ):  # text present in text1 but not in text2 (removed in text2)
            continue
        elif part.startswith(
            "+ "
        ):  # text present in text2 but not in text1 (added in text2)
            output += "+" + part[2:]
        else:
            output += part[2:]

    return output
