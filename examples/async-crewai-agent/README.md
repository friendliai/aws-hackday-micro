# Async CrewAI Agent with OpenAI-Compatible API

This example demonstrates an OpenAI-compatible chat completion API with CrewAI multi-agent function
calling capabilities. It provides both streaming and non-streaming responses with full function
calling support for complex research and analysis tasks.

## Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's chat completions endpoint
- **Function Calling**: Built-in CrewAI function tools for multi-agent workflows
- **Streaming Support**: Real-time streaming responses with function execution
- **Asynchronous Processing**: Long-running CrewAI tasks execute in the background
- **Environment-Based Configuration**: Automatic token management from environment variables
- **Task Management**: Track status and results of background tasks with unique task IDs
- **AWS AgentCore Compatible**: Follows AWS AgentCore runtime service contract patterns

## What You Need to Prepare

### 1. Friendli AI Token

- Sign up at [Friendli AI](https://friendli.ai)
- Get your API token from the dashboard
- Set it as environment variable: `export FRIENDLI_TOKEN='your_token_here'`

### 2. Python Environment

- Python 3.13 or higher
- Dependencies will be installed automatically

## How to Run

1. **Set your Friendli token:**

   ```bash
   export FRIENDLI_TOKEN='your_friendli_token_here'
   ```

1. **Install dependencies:**

   ```bash
   cd examples/async-crewai-agent
   uv sync
   ```

1. **Start the server:**

   ```bash
   uv run main.py
   ```

1. **The server will start on port 8080 at <http://0.0.0.0:8080>**

## API Endpoints

### OpenAI-Compatible API

The main endpoint for OpenAI-compatible chat completions:

#### POST `/v1/chat/completions`

This endpoint is fully compatible with OpenAI's chat completions API and includes built-in CrewAI
function tools.

**Request:**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "messages": [
      {"role": "user", "content": "Please analyze the current state of renewable energy technology"}
    ],
    "stream": true,
    "tool_choice": "auto"
  }'
```

**Built-in Function Tools:**

The API automatically includes these CrewAI function tools:

1. **`start_crewai_task`**: Start comprehensive research and analysis workflows
1. **`get_task_status`**: Check the progress of running tasks
1. **`list_all_tasks`**: Get an overview of all tasks (running, completed, failed)
1. **`get_task_result`**: Retrieve detailed results from completed tasks

**Example with Python OpenAI Client:**

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",  # Not used, but required by client
    base_url="http://localhost:8080/v1"
)

# The assistant will automatically use CrewAI tools when appropriate
response = client.chat.completions.create(
    model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    messages=[
        {"role": "user", "content": "Analyze the future of electric vehicles and create a comprehensive report"}
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### CrewAI Management API

These endpoints follow the pattern: `POST /callbacks/{endpoint_name}` and provide direct access to
CrewAI task management functions. They are designed for administrative control and monitoring of
background tasks.

**Note**: For general chat and AI assistance, use the OpenAI-compatible endpoint
`/v1/chat/completions` which automatically handles function calling when needed.

### 1. Start CrewAI Task (`/callbacks/start_crewai_task`)

Start a multi-agent CrewAI workflow in the background.

**Request:**

```bash
curl -X POST http://localhost:8080/callbacks/start_crewai_task \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Analyze the current state of renewable energy technology"
  }'
```

**Response:**

```json
{
  "task_id": "uuid-here",
  "status": "started", 
  "message": "CrewAI task has been started in the background",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### 2. Check Task Status (`/callbacks/get_task_status`)

Get the current status of a background task.

**Request:**

```bash
curl -X POST http://localhost:8080/callbacks/get_task_status \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "your_task_id_here"
  }'
```

**Response:**

```json
{
  "id": "uuid-here",
  "description": "Analyze the current state of renewable energy technology",
  "status": "completed",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:35:00Z",
  "completed_at": "2024-01-15T10:35:00Z",
  "result": "Detailed analysis results..."
}
```

### 3. Get Task Result (`/callbacks/get_task_result`)

Get the detailed results of a completed CrewAI task.

**Request:**

```bash
curl -X POST http://localhost:8080/callbacks/get_task_result \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "your_task_id_here"
  }'
```

### 4. List All Tasks (`/callbacks/list_tasks`)

Get a list of all tasks and their statuses.

**Request:**

```bash
curl -X POST http://localhost:8080/callbacks/list_tasks \
  -H "Content-Type: application/json" \
  -d '{}'
```

### 5. Health Check (`/callbacks/ping`)

Check the health status of the agent (AWS AgentCore compatible).

**Request:**

```bash
curl -X POST http://localhost:8080/callbacks/ping \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Response:**

```json
{
  "status": "Healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "active_tasks": 2,
  "total_tasks": 5
}
```

## Task Statuses

- `initializing`: Task is being set up
- `running`: CrewAI crew is actively working
- `completed`: Task finished successfully
- `failed`: Task encountered an error

## CrewAI Workflow

When you start a CrewAI task, it creates a two-agent workflow:

1. **Research Agent**: Conducts comprehensive research and analysis
1. **Writer Agent**: Transforms research into a polished report

The agents work sequentially, with the writer building upon the researcher's findings.

## AWS AgentCore Compatibility

This example follows AWS AgentCore patterns:

- Health check endpoint (`ping`)
- Asynchronous long-running task execution
- Status tracking and monitoring
- HTTP-based service contract
- Background task management

## Examples

### Using OpenAI-Compatible API

**Simple Chat:**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "messages": [
      {"role": "user", "content": "What are the benefits of solar energy?"}
    ]
  }'
```

**Streaming Chat with Function Calling:**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-235B-A22B-Instruct-2507", 
    "messages": [
      {"role": "user", "content": "Please analyze the market potential of hydrogen fuel cells and create a comprehensive report"}
    ],
    "stream": true
  }'
```

**Python Example:**

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy", 
    base_url="http://localhost:8080/v1"
)

# The AI will automatically use CrewAI tools for complex analysis
response = client.chat.completions.create(
    model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    messages=[
        {"role": "user", "content": "Research and analyze the current trends in AI automation across different industries"}
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### CrewAI Task Management Examples

**Start Background Task:**

```bash
curl -X POST http://localhost:8080/callbacks/start_crewai_task \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Comprehensive analysis of sustainable transportation options"
  }'
```

**Check Task Status:**

```bash
curl -X POST http://localhost:8080/callbacks/get_task_status \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task-uuid-from-previous-response"
  }'
```

**Get Task Results:**

```bash
curl -X POST http://localhost:8080/callbacks/get_task_result \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task-uuid-from-previous-response"
  }'
```

## Environment Setup

**Set your environment variable:**

```bash
# Linux/macOS
export FRIENDLI_TOKEN='your_friendli_token_here'

# Windows
set FRIENDLI_TOKEN=your_friendli_token_here
```

## Integration Examples

**Using with LangChain:**

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    openai_api_base="http://localhost:8080/v1",
    openai_api_key="dummy"
)
```

**Using with Anthropic-style clients:**

```python
import openai

openai.api_base = "http://localhost:8080/v1"
openai.api_key = "dummy"
```

## API Architecture

### Two Types of APIs

**1. OpenAI-Compatible Chat API (`/v1/chat/completions`)**

- General-purpose conversational AI
- Automatic function calling for complex requests
- Standard streaming/non-streaming support
- Drop-in replacement for OpenAI API

**2. CrewAI Management API (`/callbacks/*`)**

- Direct task control and monitoring
- Administrative functions for background workflows
- Detailed task status and result retrieval
- Health monitoring and system status

### Function Tools Behavior

The OpenAI-compatible assistant will automatically choose appropriate tools based on user requests:

- **Analysis/Research requests** → `start_crewai_task`
- **Status inquiries** → `get_task_status` or `list_all_tasks`
- **Result requests** → `get_task_result`

The tools work seamlessly in both streaming and non-streaming modes.

## Notes

- **Environment-based**: Uses FRIENDLI_TOKEN environment variable automatically
- **Background Processing**: CrewAI tasks run completely in the background
- **Serverless**: Uses Friendli's serverless endpoints, no GPU setup required
- **Memory Storage**: Task results stored in memory (lost on restart)
- **Production Ready**: Consider implementing persistent task storage for production
- **OpenAI Compatible**: Drop-in replacement for OpenAI chat completions API
- **Dual API Design**: Choose OpenAI API for general use, Management API for task control
