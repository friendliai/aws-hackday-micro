# AWS AI Hack Day 2025: Ready to spend a full day pushing the limits of AI?

![Banner](assets/banner.jpg)

AWS AI Hack Day Micro Conference stretches into a full-day hackathon at the AWS GenAI Loft in San
Francisco giving developers the time to go deeper, collaborate longer, and actually ship what they
start.

What’s happening on the ground:

⚡ Hands-on technical challenge powered by FriendliAI \
🧠 A 30-minute session on scaling inference and agents \
👀 An early preview of Friendli Agent shared live with the community

📍 AWS GenAI Loft, 525 Market St \
🗓️ August 22, 9:30 AM – 8:00 PM PT

If you're ready to experiment, connect, and create, this is where the Bay Area's AI community will
be. Register here → [https://lu.ma/aws-08-22-25](https://lu.ma/aws-08-22-25)

## QuickStart Guide

<a target="_blank" href="https://colab.research.google.com/github/friendliai/aws-hackday-micro/blob/main/examples/notebook/0822-hackday.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

The CLI binary is available as `fa` (alias: `friendli-app`). It manages app deploy/update and basic
suite operations.

1. Install (Python 3.11+)

- With pip:

```bash
pip install friendli-app
```

2. Authenticate

- Get a Personal Access Token: <https://friendli.ai/suite/setting/tokens>
- Export it or pass per-command with `--token`.

```bash
export FRIENDLI_TOKEN=YOUR_PAT

# Verify
fa whoami

# Or without exporting
fa --token YOUR_PAT whoami
```

3. Deploy an example app

Each app must contain a `main.py` at its root. Example apps live under `examples/`.

```bash
# Deploy examples/simple-app with a name, project, and optional env vars
fa deploy examples/simple-app \
  --name my-simple-app \
  --project-id <PROJECT_ID> \
  -e KEY1=VALUE1 -e KEY2=VALUE2
```

4. Manage apps

```bash
# List apps in a project
fa list --project-id <PROJECT_ID>

# Update source archive from a directory
fa update <APP_ID> ./path/to/app

# Restart or terminate
fa restart <APP_ID>
fa terminate <APP_ID>
```

Notes

- Command alias: `friendli-app` is identical to `fa`.
- Example deployment prints a link to view status in the Suite UI.

## Commands

Global options

- `--token`: Personal access token; overrides `FRIENDLI_TOKEN`.
- `-h, --help`: Show help for the current command.

Common

- `fa whoami`: Show logged-in user info.
  - Usage: `fa whoami` or `fa --token <PAT> whoami`
- `fa version`: Show CLI version.
  - Usage: `fa version`

Apps

- `fa deploy <APP_DIR>`: Deploy an app directory.
  - Options: `-n, --name <NAME>`, `-p, --project-id <PROJECT_ID>`, `-e, --env KEY=VALUE`
    (repeatable)
  - Notes: `main.py` must exist at app root; ~50MB directory limit; detects `pyproject.toml` or
    `requirements.txt` to bundle deps.
  - Example: `fa deploy examples/simple-app -n my-simple-app -p <PROJECT_ID> -e KEY1=VALUE1`
- `fa update <APP_ID> <APP_DIR>`: Update an app’s source archive.
  - Notes: `main.py` required; ~50MB limit.
  - Example: `fa update <APP_ID> ./my-app`
- `fa list --project-id <PROJECT_ID>`: List apps in a project.
  - Example: `fa list -p <PROJECT_ID>`
- `fa restart <APP_ID>`: Restart an app.
  - Example: `fa restart <APP_ID>`
- `fa terminate <APP_ID>`: Terminate an app.
  - Example: `fa terminate <APP_ID>`

## Example Apps

Each example is its own Python project. See the example’s README for setup, dependencies, and usage.

- examples/simple-app: Minimal AgentApp with sync/async callbacks and streaming.
  [README](examples/simple-app/README.md)
- examples/streaming-chat-memory: Streaming chat with persistent memory (mem0), OpenAI-compatible
  `/v1/chat/completions`. [README](examples/streaming-chat-memory/README.md)
- examples/daily-assistant-mcp: MCP server exposing practical tools (tip calc, timezone, BMI,
  password). [README](examples/daily-assistant-mcp/README.md)
- examples/debug-echo: Tiny FastAPI echo service for connectivity testing.
  [README](examples/debug-echo/README.md)
- examples/debug-fai: FastAPI app calling Friendli Serverless via OpenAI SDK; includes passthrough
  endpoint. [README](examples/debug-fai/README.md)
- examples/langgraph-research-agent: LangGraph multi-agent research workflow with streaming.
  [README](examples/langgraph-research-agent/README.md)
- examples/async-crewai-agent: CrewAI-based background task agent with progress and results
  endpoints. [README](examples/async-crewai-agent/README.md)
- examples/adk-multi-agent-research: Google ADK-style multi-agent research FastAPI service.
  [README](examples/adk-multi-agent-research/README.md)
- examples/autogen-dev-team: AutoGen multi-agent dev team orchestrating design→code→review.
  [README](examples/autogen-dev-team/README.md)

## SDK Guide

Build lightweight HTTP agents using the SDK in `friendli_app.sdk`.

- Import: `from friendli_app.sdk import AgentApp`
- Define callbacks with `@app.callback` (sync, async, or generators for streaming)
- Run locally: `python main.py` (uses Uvicorn under the hood)
- Invoke: `POST /callbacks/{callback_name}` with JSON body

Example

```python
import asyncio
from friendli_app.sdk import AgentApp

app = AgentApp()

@app.callback
def greet(name: str = "World"):
    return {"message": f"Hello, {name}!"}

@app.callback
async def greet_async(name: str = "World"):
    await asyncio.sleep(1)
    return {"message": f"Hello, {name}! (async)"}

@app.callback
def stream(n: int = 3):
    for i in range(n):
        yield {"i": i, "msg": f"chunk {i+1}/{n}"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
```

Invoke callbacks

- JSON response:

```bash
curl -s -X POST http://localhost:8080/callbacks/greet \
  -H 'Content-Type: application/json' \
  -d '{"name": "Ada"}'
```

- Streaming (SSE):

```bash
curl -N -X POST http://localhost:8080/callbacks/stream \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -d '{"n": 5}'
```

Notes

- The request body JSON is mapped directly to the callback function parameters.
- Generator or async-generator callbacks stream Server-Sent Events (`text/event-stream`).
- To deploy an SDK app with the CLI, ensure your project root contains `main.py` with an `AgentApp`
  instance.
