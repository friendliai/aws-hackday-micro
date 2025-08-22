# Streaming Chat with Memory

A streaming chat agent with persistent memory capabilities using
[mem0](https://github.com/mem0ai/mem0) for memory management and
[Friendli Serverless Endpoints](https://friendli.ai/docs/guides/serverless_endpoints/text-generation)
for LLM inference.

## Overview

This example demonstrates how to build an AI agent that:

- 🧠 **Remembers conversations** across sessions using mem0
- 🚀 **Streams responses** in real-time for better user experience
- ⚡ **Uses Friendli serverless** for fast, scalable LLM inference
- 🔄 **Follows AWS AgentCore patterns** for compatibility
- 🌐 **Provides HTTP API** compatible with AgentCore Runtime

The agent maintains conversational context, learns user preferences, and provides personalized
responses based on chat history.

## Features

- **🔗 OpenAI API Compatibility**: Drop-in replacement for OpenAI API with `/v1/chat/completions`
- **🧠 Persistent Memory**: Stores and retrieves conversation history per user using mem0
- **🚀 Real-time Streaming**: SSE-based streaming responses with `stream` parameter
- **⚡ Multiple Endpoints**: OpenAI-compatible and custom callback endpoints
- **🔧 Memory Management**: View and clear user memories via API
- **🩺 Health Monitoring**: Built-in health check endpoint
- **🛡️ Error Handling**: Comprehensive error handling and logging
- **🎯 Framework Compatible**: Works with any OpenAI-compatible client or framework

## Prerequisites

### Required Tokens

You need a **Friendli API token** to use Friendli serverless endpoints:

1. Sign up at [Friendli Platform](https://friendli.ai/)
1. Navigate to your dashboard and generate an API token
1. Set the token as an environment variable:

```bash
export FRIENDLI_TOKEN=dummy_api_key_here
```

You also need an **OpenAI API key** if you want to use examples that rely on OpenAI embeddings:

```bash
export OPENAI_API_KEY=sk-xxxxxxxx
```

### System Requirements

- Python 3.13 or higher
- At least 1GB RAM (for mem0 memory storage)
- Network access to Friendli API endpoints

## Installation

1. **Set your tokens:**

```bash
export FRIENDLI_TOKEN=dummy_api_key_here
export OPENAI_API_KEY=sk-xxxxxxxx
```

2. **Install dependencies:**

```bash
cd examples/streaming-chat-memory
uv sync
```

## Usage

### Start the Server

```bash
uv run main.py
```

The server will start on `http://0.0.0.0:8080` and display available endpoints.

### OpenAI-Compatible API

You can use this agent as a drop-in replacement for OpenAI API by setting the base URL to
`http://0.0.0.0:8080/v1`

> ⚠️ **Note**: If you are using OpenAI embeddings within this project, make sure you have\
> \`OPENAI_API_KEY\` exported in your shell.

#### Using OpenAI Python SDK

```python
from openai import OpenAI

# Initialize client with your Friendli token
client = OpenAI(
    api_key="dummy_api_key",
    base_url="http://0.0.0.0:8080/v1"
)

# Non-streaming chat
response = client.chat.completions.create(
    model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    messages=[
        {"role": "user", "content": "I love cooking Italian food!"}
    ],
    user="user123",  # For memory management
    temperature=0.7
)
print(response.choices[0].message.content)

# Streaming chat
stream = client.chat.completions.create(
    model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    messages=[
        {"role": "user", "content": "What pasta recipe would you recommend?"}
    ],
    user="user123",
    stream=True,
    temperature=0.7
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

#### Using cURL

**Non-streaming:**

```bash
curl -X POST http://0.0.0.0:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "messages": [
      {"role": "user", "content": "Hello! I love cooking Italian food."}
    ],
    "user": "user123",
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

**Streaming:**

```bash
curl -X POST http://0.0.0.0:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "messages": [
      {"role": "user", "content": "What pasta recipe would you recommend?"}
    ],
    "user": "user123",
    "stream": true,
    "temperature": 0.7
  }'
```

#### Using with LangChain

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Initialize LangChain with your agent
llm = ChatOpenAI(
    api_key="dummy_api_key",
    base_url="http://0.0.0.0:8080/v1",
    model="Qwen/Qwen3-235B-A22B-Instruct-2507"
)

# Chat with memory
response = llm.invoke([
    HumanMessage(content="I love cooking Italian food!")
])
print(response.content)
```

### Utility Endpoints

For memory management and health monitoring, you can use these utility endpoints:

#### Health Check

```bash
curl -X POST http://0.0.0.0:8080/callbacks/health
```

#### Get User Memories

```bash
curl -X POST http://0.0.0.0:8080/callbacks/get_memories \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "limit": 10
  }'
```

#### Clear User Memories

```bash
curl -X POST http://0.0.0.0:8080/callbacks/clear_memories \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123"
  }'
```

## Configuration

### Model Selection

You can use any model from Friendli's catalog with the OpenAI-compatible API:

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy_api_key",
    base_url="http://0.0.0.0:8080/v1"
)

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-0528",  # Different model
    messages=[{"role": "user", "content": "Hello!"}],
    user="user123"
)
```

Available models include:

- `Qwen/Qwen3-235B-A22B-Instruct-2507` (default)
- `deepseek-ai/DeepSeek-R1-0528`
- `skt/A.X-4.0`
- And more! See
  [Friendli's model catalog](https://friendli.ai/docs/guides/serverless_endpoints/text-generation)

### Generation Parameters

Standard OpenAI parameters are supported:

- `max_tokens`: Maximum tokens to generate (default: 1000)
- `temperature`: Creativity level, 0.0-1.0 (default: 0.7)
- `stream`: Enable streaming responses (default: false)
- `user`: User identifier for memory management (OpenAI standard)
- `user_id`: Alternative user identifier (custom field, backwards compatibility)

## Memory System

The example uses [mem0](https://github.com/mem0ai/mem0) for intelligent memory management:

- **Automatic Memory**: Conversations are automatically stored
- **Semantic Search**: Retrieves relevant memories based on context
- **User Isolation**: Each user has separate memory space
- **Persistent Storage**: Memories persist across server restarts

### Memory Workflow

1. **Query Processing**: User message is analyzed for relevant memories
1. **Memory Retrieval**: System finds related past conversations
1. **Context Building**: Relevant memories are added to system prompt
1. **Response Generation**: LLM generates contextually aware response
1. **Memory Storage**: New conversation is stored for future use

## Framework Compatibility

This agent is compatible with multiple frameworks and standards:

### AWS AgentCore Compatibility

- **HTTP Protocol**: Runs on standard HTTP port 8080
- **Streaming Support**: Implements SSE-based streaming responses
- **Error Handling**: Proper HTTP status codes and error responses
- **JSON API**: All endpoints use JSON for request/response
- **Health Checks**: Built-in health monitoring endpoint

### OpenAI API Compatibility

- **Standard Endpoints**: `/v1/chat/completions` endpoint
- **Request/Response Format**: Follows OpenAI API specification
- **Streaming Protocol**: SSE with `data:` prefixed JSON chunks
- **Error Format**: OpenAI-compatible error responses
- **Parameter Support**: Standard OpenAI parameters (model, temperature, max_tokens, etc.)

### Framework Support

Works out-of-the-box with:

- **OpenAI Python SDK**: Drop-in replacement
- **LangChain**: Use `ChatOpenAI` with custom `base_url`
- **LlamaIndex**: Compatible with OpenAI LLM wrapper
- **Autogen**: Works with OpenAI-compatible models
- **Any OpenAI-compatible client**: Just change the `base_url`

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest
```

### Local Development

For development, you can run with hot reload:

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

## Troubleshooting

### Common Issues

1. **"FRIENDLI_TOKEN not found"**

   - Make sure you've set the environment variable: `export FRIENDLI_TOKEN=your_token`

1. **Memory errors**

   - Ensure you have sufficient RAM and disk space for mem0 storage

1. **Model not found**

   - Check the
     [Friendli model catalog](https://friendli.ai/docs/guides/serverless_endpoints/text-generation)
     for available models

1. **Connection errors**

   - Verify network access to `https://api.friendli.ai/serverless/v1`

### Debug Mode

Enable debug logging by setting:

```bash
export LOG_LEVEL=DEBUG
python main.py
```

## Architecture

```
OpenAI-Compatible Client
        ↓
/v1/chat/completions (OpenAI API)
        ↓
AgentApp (FastAPI + Custom Callbacks)
        ↓
Memory Lookup (mem0) → Context Injection
        ↓
Friendli Serverless API (LLM Inference)
        ↓
Streaming Response (SSE) ← Memory Storage
        ↓
Client (OpenAI SDK, LangChain, etc.)
```

### Request Flow

1. **Client Request**: Any OpenAI-compatible client sends request to `/v1/chat/completions`
1. **Memory Lookup**: System searches relevant memories using mem0
1. **Context Building**: Memories are injected into system prompt
1. **LLM Inference**: Friendli serverless processes the enhanced prompt
1. **Response Streaming**: Real-time SSE streaming back to client
1. **Memory Storage**: Conversation is stored for future context

## Contributing

1. Fork the repository
1. Create a feature branch
1. Make your changes
1. Add tests
1. Submit a pull request

## License

This example is part of the Friendli Agent SDK and follows the same license terms.

## Links

- [Friendli Platform](https://friendli.ai/)
- [Friendli Serverless Documentation](https://friendli.ai/docs/guides/serverless_endpoints/text-generation)
- [mem0 Documentation](https://docs.mem0.ai/)
- [AWS AgentCore Documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/)
