"""
HuggingFace Multi-worker Server Worker

This module loads LLM and serves API with single endpoint to allow :module:`backend` to request
LLM for inference. This module is designed to have multiple instances of it running all on same
host without advanced cloud-native concepts. This is part of a PoC to allow multi-tenant inference
of HuggingFace SDK
"""

import asyncio
import json
import logging
from typing import TYPE_CHECKING

import requests
import torch
import transformers
from aiohttp import web
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from aiohttp.web import Request, Response


# This is a PoC. Log everything for observability
logging.basicConfig(level=logging.DEBUG)

# Constants
MODEL = "tiiuae/falcon-40b-instruct"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
PIPE = transformers.pipeline(
    "conversational",
    model=MODEL,
    tokenizer=TOKENIZER,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)


def answer_question(article: str, question: str) -> str:
    """
    Logic to infer LLM to answer `question` assuming its knowledge is all from `article`

    :param str article:
        Content to be used as knowledge source to answer `question`
    :param str question:
        Question to be answered by LLM via inference in :module:`worker` instance
    :return:
        LLM response
    """

    PROMPT = (
        "You are an artificial intelligence assistant. The assistant gives helpful,"
        " detailed, and polite answers to the user's questions. Your only source of"
        f' knowledge is following article between quote marks "{article}"\n{question}'
    )

    conversation = transformers.Conversation(PROMPT)

    conversation = PIPE(
        conversation,
        top_k=10,
        temperature=0.45,
        use_cache=True,
        max_length=3000,
    )
    return conversation.generated_responses[-1]


async def handle_post(request: "Request") -> "Response":
    """
    Logic for 'POST /' endpoint

    This endpoint is exposed to :module:`backend` to route requests for LLM to current
    :module:`worker` instance

    :param :class:`Request` request:
        Provided by :module:`aiohttp`
    :return:
        :class:`Response` object
    """

    values = await request.json()

    answer = answer_question(values["article"], values["question"])

    return web.Response(text=json.dumps({"answer": answer}))


if __name__ == "__main__":
    # REF: https://stackoverflow.com/a/45253163
    # Build and start :module:`aiohttp` :class:`Application`. Start :class:`Application` with `port`
    # set to '0' to allow :module:`aiohttp` to pick any available port for binding. Then use
    # suggested method from SO to get `port`
    app = web.Application()
    app.add_routes(
        [
            web.post("/", handle_post),
        ]
    )

    loop = asyncio.get_event_loop()
    # continue server bootstraping
    handler = app.make_handler()
    coroutine = loop.create_server(handler, "127.0.0.1", 0)
    server = loop.run_until_complete(coroutine)
    port = server.sockets[0].getsockname()[1]
    print(f"Serving on http://{':'.join(server.sockets[0].getsockname())}")

    # Register instance with :module:`backend`. This is naive service discovery
    requests.put("http://127.0.0.1:8080/workers", json={"port": port}, timeout=15)

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        requests.delete(f"http://127.0.0.1:8080/workers/{port}", timeout=15)
        server.close()
        loop.run_until_complete(server.wait_closed())
        loop.close()
