# HuggingFace Multi-worker Server

Python nature brings a lot of challenges when dealing with blocking IO. HuggingFace SDK doesn't provide an out-of-box solution to having inference on models be threaded, although the lower-level structures (PyTorch and Tensorflow) provides the necessary tooling. [HF docs suggest using multi-threaded web server](https://huggingface.co/docs/transformers/main/pipeline_webserver), but my attempts didn't to apply the same snippet didn't resolve well.

As I needed an urgent PoC of being able to provide multi-tenant (more than one user using using LLM capabilities at once) service. I decided to build a PoC that follows workers concept, where multiple number of [`workers`](./worker.py) can be started alongside [`backend`](./backend.py) to provide multi-tenant API for LLM inference.

This specific demo runs `falcon-40b-instruct` model in `conversational` mode, and allows users to provide knowledge source, `article`, and ask `question` so LLM would answer it assuming its only knowledge is `article`. To use this PoC:
1. Create a `venv`, and activate it:
```bash
python[3[.11]] -m venv venv
source venv/bin/activate
```
2. Install runtime dependencies:
```bash
pip install .
```
3. Start `backend`:
```bash
python backend.py
```
4. Start one or multiple `worker` instances: This is doable by starting new shell and sourcing earlier created `venv` and starting the `worker` instance:
```bash
# New shell, Working directory is this project
source venv/bin/active
python worker.py
```
5. Make a request to `backend`:
```bash
curl -X POST -H "Content-type: application/json" -d '{"article": "Today is Wed. 21st. Jun 2023. The weather is hot. I am currently not at home, but at office. I am working on implementing multi-threading for the LLM backend", "question":"What date is it?"}' 'http://127.0.0.1:8080'

# The date mentioned in the article is 21st. June...
```

## On the road to MVP
To Convert this into an MVP, following points should be tackled:
1. Add health check to `backend`: Upon making request to `worker` instance, if `worker` instance is unreachable over period of retries, it should be removed from `backend` registered workers.
2. Add access-control on `backend` endpoints: Endpoints of `backend` for registering and de-registering `worker` instances should be scoped-down to prevent misuse.
3. Return inference time with requests for analytics.
4. Containerize PoC to run with `docker-compose`.
