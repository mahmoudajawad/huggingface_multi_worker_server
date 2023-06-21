"""
HuggingFace Multi-worker Server Backend

This module serves an API with two functionalities:
- Register and de-register :module:`worker` instances
- Ask LLM to answer question from given article

The web server is provided by :module:`aiohttp`, and requests to LLM are passed to workers
registered via naive locking mechanism. If no worker is available to handle a request, server would
response with error '503'. Else, a thread is started that wraps a :module:`requests`.`post` request
to allow server to continue to serve its functions without having block IO until receiving the
response from :module:`worker`
"""

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Optional, TypedDict

import requests
from aiohttp import web

if TYPE_CHECKING:
    from aiohttp.web import Request, Response


# This is a PoC. Log everything for observability
logging.basicConfig(level=logging.DEBUG)


# Type hints
class Worker(TypedDict):
    """
    Type-hint for `WORKERS` constant item
    """

    locked_since: Optional[float]
    result: Optional[tuple[int, "WorkerResult"]]


class WorkerResult(TypedDict):
    """
    Type-hint for :class:`Worker`.`result[1]` :class:`dict`
    """

    answer: str


# Constants
WORKERS: dict[str, "Worker"] = {}


async def handle_put_workers(request: "Request") -> "Response":
    """
    Logic for 'PUT /workers' endpoint

    This endpoint is exposed to :module:`worker` instances to register themselves so
    :module:`backend` would request them for inference requests

    :param :class:`Request` request:
        Provided by :module:`aiohttp`
    :return:
        :class:`Response` object
    """

    payload = await request.json()

    WORKERS[str(payload["port"])] = {"locked_since": None, "result": None}

    return web.Response(status=201)


async def handle_delete_workers_port(request):
    """
    Logic for 'DELETE /workers/{port}' endpoint

    This endpoint is exposed to :module:`worker` instances to de-register themselves upon shutting
    down

    :param :class:`Request` request:
        Provided by :module:`aiohttp`
    :return:
        :class:`Response` object
    """

    port = request.match_info.get("port")

    del WORKERS[port]

    return web.Response(status=201)


def request_answer_question(
    article: str, question: str, port: str, worker: "Worker"
) -> None:
    """
    Wrapper for :module:`requests`.`post` request to :module:`worker` instance

    This thin wrapper acts as `target` to :class:`Thread` to allow web server to make multiple
    requests to :module:`worker` instances without blocking IO

    :param str article:
        Content to be used as knowledge source to answer `question`
    :param str question:
        Question to be answered by LLM via inference in :module:`worker` instance
    :param str port:
        Port number of :module:`worker` instance to request
    :param :class:`Worker` worker:
        :class:`Worker` object. This value is passed so `result` item is set per request response
    """

    r = requests.post(
        f"http://127.0.0.1:{port}",
        json={"article": article, "question": question},
        timeout=180,
    )

    if r.status_code != 200:
        worker["result"] = (
            r.status_code,
            {"answer": "I could not answer your question. Try again later."},
        )

    worker["result"] = (200, r.json())


async def handle_post_messages(request: "Request") -> "Response":
    """
    Logic for 'POST /messages' endpoint

    This endpoint is exposed to applications to ask questions on knowledge, all provided in single
    request

    :param :class:`Request` request:
        Provided by :module:`aiohttp`
    :return:
        :class:`Response` object
    """

    # Get payload
    payload = await request.json()

    # Find any available worker to request
    for port, args in WORKERS.items():
        if args["locked_since"] is None:
            args["locked_since"] = time.time()
            args["result"] = None
            break
    else:
        # If none, return '503' with descriptive message
        return web.Response(
            status=503,
            text=(
                "Service Unavailable: All workers are busy serving other requests. Try"
                " again later"
            ),
        )

    # As we have a worker to request, start a thread with :func:`request_answer_question`
    thread = threading.Thread(
        target=request_answer_question,
        args=(payload["article"], payload["question"], port, args),
    )
    thread.start()

    # Wait until we get result
    while True:
        await asyncio.sleep(0.5)
        if args["result"]:
            break

    # Release naive lock
    args["locked_since"] = None
    answer = args["result"][1]["answer"]

    return web.Response(text=answer)


if __name__ == "__main__":
    # Build and start :module:`aiohttp` :class:`Application`
    app = web.Application()
    app.add_routes(
        [
            web.put("/workers", handle_put_workers),
            web.delete("/workers/{port}", handle_delete_workers_port),
            web.post("/messages", handle_post_messages),
        ]
    )
    web.run_app(app)
