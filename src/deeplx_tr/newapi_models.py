"""
Collect all models for newapi @acone:3001 in one place.

Use from pkgutil import resolve_name
resolve_name("newapi_models:newapi_models")

"""
from pathlib import Path

import diskcache

cache = diskcache.Cache(Path.home() / ".diskcache" / "newapi-tr")

# str or tuple, tuple is hashable, to collect stats (defaultdict) later on
models = [
    ("gpt-3.5-turbo", "https://duck2api-lovat.vercel.app", "YourAPIKey"),
    ("gpt-3.5-turbo", "https://duck2api-u3l3laz3.b4a.run", "YourAPIKey"),
    ("gpt-3.5-turbo", "https://mikeee-duck2api.hf.space/hf", "YourAPIKey"),
    ("gpt-3.5-turbo", "https://sun-i-duck2api.hf.space", "YourAPIKey"),
    ("gpt-3.5-turbo", "https://Raimbault-duck2api.hf.space", "YourAPIKey"),
    ("gpt-3.5-turbo", "https://gyxz-duck2api.hf.space", "YourAPIKey"),
    # google cloud run test
    "gpt-3.5-turbo-grun",
    "gpt-4o-grun",
    # chat2api at acone:3007 docker
    "gpt-3.5-turbo-chat2api",
    "gpt-4o-chat2api",
    "gpt-4-chat2api",
    # acone:5005 -> oracle2:5005
    "gpt-3.5-turbo-oracle2",
    "gpt-4o-oracle2",
    # acone:25005 -> oraclex:5005 is having problem to start ubuntu@oracle:~/chat2api-lanqian, fix: disable kubelet restart
    "gpt-3.5-turbo-oracle",
    "gpt-4o-oracle",
    # "gpt-3.5-turbo",  # duck2api at acone:8080 docker
    "gpt-3.5-turbo-d2a",
    # "claude-3-haiku-d2a",
    # "llama-3-70b-d2a",
    # "mixtral-8x7b-d2a",
]

# redteam qwen hf
models += ['qwen-turbo-hf',
 'qwen-max-hf',
 'qwen-max-longcontext-hf',
 'qwen-plus-hf',
 'qwen-vl-max-hf',
 'qwen-vl-plus-hf',
 'qwen-v1-hf',
 'qwen-v1-vision-hf']

# curl -XPOST https://mikeee-reka.hf.space/hf/v1/chat/completions -H "Authorization: Bearer 316287378"  -H "Content-Type: application/json" --data "{\"model\": \"reka-core\", \"messages\": [{\"role\": \"user\", \"content\": \"Say this is a test!\"}]}"
# newapi at acone via mikeee-reka.hf.space free reka.ai
models += [
    # "reka-core",  # stream OK, need fix
    # "reka-edge",
    # "reka-flash",
]

# https://api.deepseek.com , disabled in newapi
models += [
    # "deepseek-chat",
    # "deepseek-coder",
]

# llm-eradteam-deepseek @ acone:3003 via newapi @ acone
# docker deepseek-free-api
# newapi http://142.171.18.103:3003
models += [
    "deepseek-chat",
    # "deepseek-coder",
]
models += [
    "deepseek-chat-hf",
    "deepseek-coder-hf",
]

# llm-red-team acone:3004 kimi native node @ acone pm2 kimi-free-api
# need 3 tries, takes too long, 3x6s
models += [
    # "moonshot-v1",
    # "moonshot-v1-8k",
    # "moonshot-v1-32k",
    # "moonshot-v1-128k",
    # "moonshot-v1-vision",
]

# llm red team glm acone pm2 acone:3005
models += [
    "glm-3-turbo",
    "glm-4",
    "glm-4v",
    "glm-v1",
    "glm-v1-vision",
]

# groq2api acone:3006
models += [
    "llama3-70b-8192-groq",
    "mixtral-8x7b-32768-groq"
    # 'llama3-8b-8192-groq',
    # 'gemma-7b-it-groq',
]

# from check_models.py
models += [
    "gpt-3.5-turbo-uuci",
    "gpt-4-turbo-uuci",
    "gpt-4-uuci",
    "claude-3.5-sonnet-uuci",
    "gpt-4o-furry",
    "gpt-4-turbo-2024-04-09-furry",
    "gpt-3.5-turbo-furry",
    # "gpt-4-0o0",
    # "gpt-4o-0o0",
    # "gpt-4-turbo-0o0",
    # "claude-3-opus-20240229-0o0",  # 速率限制错误 令牌被上游接口限流
    # "claude-3-sonnet-20240229-0o0",
    # "claude-3-haiku-20240307-0o0",
    # "claude-2.0-0o0",
    # "zephyr-0o0",
    # "kimi-0o0",
    # "kimi-vision-0o0",
    # "reka-core-0o0",
    # "reka-edge-0o0",
    # "reka-flash-0o0",
    # "command-r-0o0",
    # "command-r-plus-0o0",
    # "deepseek-chat-0o0",
    # "deepseek-coder-0o0",
    # "google-gemini-pro-0o0",
    # "gemma-7b-it-0o0",
    # "llama2-7b-2048-0o0",
    # "llama2-70b-4096-0o0",
    # "llama3-8b-8192-0o0",
    # "llama3-70b-8192-0o0",
    # "mixtral-8x7b-32768-0o0",
    # "gpt-3.5-turbo-hf",
    # "gpt-3.5-turbo-edwagbb",
    # "gpt-3.5-turbo-smgc",
    # "gpt-3.5-turbo-sfun",
    # "gpt-3.5-turbo-neuroama",
]

# ##################
# direct api-key, no use registration
# models += [
models_xxxx = [
    *{
        "gpt-3.5-turbo-mihoyo": "gpt-3.5-turbo",
        "reka-core-mihoyo": "reka-core",
        "reka-edge-mihoyo": "reka-edge",
        "reka-flash-mihoyo": "reka-flash",
        "deepseek-chat-mihoyo": "deepseek-chat",
        # "deepseek-coder-mihoyo": "deepseek-coder",
        "kimi-mihoyo": "kimi",
        "kimi-vision-mihoyo": "kimi-vision",
        "glm-4-mihoyo": "glm-4",
    }
]

# models = ["kimi-0o0"]
# models = [*{"Llama3-8b-8192-sgai": "Llama3-8b-8192", "Llama3-70b-8192-sgai": "Llama3-70b-8192"}]

models += [
    "gpt-3.5-turbo-bh",
    "gpt-4-turbo-preview-bh",
    "gpt-4-bh",
    "gpt-4o-bh",
    "gpt-3.5-turbo-instruct-bh",
]

# https://linux.do/t/topic/63908
models2 = [
    *{
        "gpt-3.5-turbo-zhc": "gpt-3.5-turbo",
        "kimi-zhc": "kimi",
        "glm4-zhc": "glm4",
        "gpt-4-zhc": "gpt-4",
    }
]

# # does not work, need to modify v1/chat -> hf/v1/chat?
# models += ["deepseek-chat-hf", "deepseek-coder-hf"]

# -bf
models += [
    # 'gpt-3.5-turbo-bf',
    # 'command-bf', 'command-light-bf', 'command-light-nightly-bf',
    "command-nightly-bf",
    "command-r-bf",
    "command-r-plus-bf",
    "deepseek-chat-bf",
    "deepseek-coder-bf",
    "glm-3-turbo-bf",
    # 'glm-4-bf', 'glm-4-flash-bf', 'glm-4v-bf'
]

# zapi file:///C:\mat-dir\snippets\zapi-models.py
# "https://api.953959.xyz" http://acone:3001
models += [
    "gpt-3.5-turbo-z",
    "gpt-4-turbo-z",
]

# models = ["gpt-3.5-turbo-d2a"]


# https://linux.do/t/topic/98814 https://new.oaifree.com/ token: https://chatgpt.com/api/auth/session ...G2Ow

newapi_models = models[:]

# cache.set("models", models)
