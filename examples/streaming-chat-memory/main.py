"""
Streaming Chat with Memory Example using mem0 and Friendli Serverless

This example demonstrates how to create a streaming chat agent with persistent memory
using mem0 for memory management and Friendli serverless endpoints for LLM inference.

The agent maintains conversational context across sessions and can recall user preferences
and past interactions.
"""

import json
import os
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from mem0 import Memory
from openai import OpenAI
from pydantic import BaseModel, Field

from friendli_app.sdk import AgentApp

# Initialize the AgentApp
app = AgentApp()

# Initialize OpenAI client for Friendli serverless endpoints
client = OpenAI(
    api_key=os.getenv("FRIENDLI_TOKEN"),
    base_url="https://api.friendli.ai/serverless/v1",
)

# Initialize mem0 memory
memory = Memory()

# Default model to use
DEFAULT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"


# OpenAI-compatible Pydantic models
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: list[ChatMessage]
    max_tokens: int | None = 1000
    temperature: float | None = 0.7
    stream: bool | None = False
    user: str | None = None  # OpenAI user field for memory/user_id
    # Custom field for memory user_id (backwards compatibility)
    user_id: str | None = Field(default=None, description="Custom user identifier for memory")


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage | None = None
    delta: ChatMessage | None = None
    finish_reason: str | None = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage | None = None


def _timestamp() -> str:
    """Generate ISO timestamp."""
    return datetime.now(UTC).isoformat()


def get_relevant_memories(query: str, user_id: str, limit: int = 5) -> list[dict]:
    """Retrieve relevant memories for the user query."""
    try:
        relevant_memories = memory.search(query=query, user_id=user_id, limit=limit)
        return relevant_memories.get("results", [])
    except Exception as e:
        print(f"Error retrieving memories: {e}")
        return []


def add_conversation_to_memory(messages: list[dict], user_id: str) -> None:
    """Add conversation to memory."""
    try:
        memory.add(messages, user_id=user_id)
    except Exception as e:
        print(f"Error adding to memory: {e}")


def build_system_prompt(memories: list[dict]) -> str:
    """Build system prompt with relevant memories."""
    base_prompt = """You are a helpful AI assistant with memory capabilities.
You can remember past conversations and user preferences to provide personalized assistance.
Use the provided memories to give contextually relevant responses."""

    if memories:
        memories_str = "\n".join([f"- {mem['memory']}" for mem in memories])
        return f"{base_prompt}\n\nRelevant memories:\n{memories_str}"

    return base_prompt


@app.callback
def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": _timestamp(), "service": "streaming-chat-memory"}


@app.callback
def get_memories(user_id: str = "default_user", limit: int = 10):
    """
    Retrieve memories for a specific user.

    Args:
        user_id: User identifier
        limit: Maximum number of memories to retrieve
    """
    try:
        memories = memory.get_all(user_id=user_id, limit=limit)
        return {
            "user_id": user_id,
            "memories": memories,
            "count": len(memories),
            "timestamp": _timestamp(),
        }
    except Exception as e:
        return {"error": str(e), "timestamp": _timestamp()}


@app.callback
def clear_memories(user_id: str):
    """
    Clear all memories for a specific user.

    Args:
        user_id: User identifier
    """
    try:
        # Get all memories for the user first
        all_memories = memory.get_all(user_id=user_id)

        # Delete each memory
        deleted_count = 0
        for mem in all_memories:
            memory.delete(memory_id=mem["id"])
            deleted_count += 1

        return {"user_id": user_id, "deleted_count": deleted_count, "timestamp": _timestamp()}
    except Exception as e:
        return {"error": str(e), "timestamp": _timestamp()}


@app.get("/")
def root():
    """Root endpoint with service information."""
    return {
        "name": "Streaming Chat with Memory",
        "description": "OpenAI-compatible streaming chat with persistent memory using mem0 and FriendliAI",
        "version": "1.0.0",
        "endpoints": {
            "/v1/chat/completions": "POST - OpenAI-compatible chat with streaming and memory",
            "/callbacks/health": "POST - Health check",
            "/callbacks/get_memories": "POST - Retrieve user memories",
            "/callbacks/clear_memories": "POST - Clear user memories",
        },
        "features": [
            "OpenAI-compatible API",
            "Streaming responses",
            "Persistent conversation memory with mem0",
            "User-specific memory contexts",
            "Memory-enhanced responses",
        ],
        "openai_compatibility": {
            "base_url": "http://localhost:8080/v1",
            "supports_streaming": True,
            "user_identification": "user or user_id field in request",
        },
        "model": {"default": DEFAULT_MODEL, "provider": "FriendliAI"},
        "environment_required": ["FRIENDLI_TOKEN"],
        "timestamp": _timestamp(),
        "status": "healthy",
    }


def get_user_id_from_request(request: ChatCompletionRequest) -> str:
    """Extract user ID from request, with fallback logic."""
    return request.user_id or request.user or "default_user"


def create_openai_response(
    completion_id: str, model: str, content: str, finish_reason: str = "stop"
) -> ChatCompletionResponse:
    """Create OpenAI-compatible response."""
    return ChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=int(datetime.now(UTC).timestamp()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=0,  # Simplified for now
            completion_tokens=0,
            total_tokens=0,
        ),
    )


def create_stream_chunk(
    completion_id: str, model: str, content: str = "", finish_reason: str | None = None
) -> str:
    """Create streaming chunk in OpenAI format."""
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(datetime.now(UTC).timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


# Add OpenAI-compatible endpoint
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> Any:
    """
    OpenAI-compatible chat completions endpoint.
    Supports both streaming and non-streaming based on the 'stream' parameter.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    # Get user message (last user message)
    user_message = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    user_id = get_user_id_from_request(request)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    try:
        # Get relevant memories
        relevant_memories = get_relevant_memories(user_message, user_id)

        # Build system prompt with memories
        system_prompt = build_system_prompt(relevant_memories)

        # Prepare messages for Friendli API
        messages = []

        # Add system message if we have memories or if there's already a system message
        if system_prompt or any(msg.role == "system" for msg in request.messages):
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation messages (skip system messages from request to avoid duplication)
        for msg in request.messages:
            if msg.role != "system":
                messages.append({"role": msg.role, "content": msg.content})

        # Handle streaming vs non-streaming
        if request.stream:
            return StreamingResponse(
                stream_openai_response(
                    completion_id, request.model, messages, user_message, user_id, request
                ),
                media_type="text/event-stream",
            )
        else:
            # Non-streaming response
            response = client.chat.completions.create(
                model=request.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=False,
            )

            assistant_response = response.choices[0].message.content

            # Add to memory
            memory_messages = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_response},
            ]
            add_conversation_to_memory(memory_messages, user_id)

            return create_openai_response(completion_id, request.model, assistant_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


async def stream_openai_response(
    completion_id: str,
    model: str,
    messages: list[dict],
    user_message: str,
    user_id: str,
    request: ChatCompletionRequest,
):
    """Generate streaming response in OpenAI format."""
    try:
        # Start streaming
        response_chunks = []
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                response_chunks.append(content)

                # Yield streaming chunk
                yield create_stream_chunk(completion_id, model, content)

        # Final chunk with finish_reason
        yield create_stream_chunk(completion_id, model, finish_reason="stop")

        # End stream
        yield "data: [DONE]\n\n"

        # Add conversation to memory
        full_response = "".join(response_chunks)
        memory_messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": full_response},
        ]
        add_conversation_to_memory(memory_messages, user_id)

    except Exception as e:
        # Error chunk
        error_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(datetime.now(UTC).timestamp()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            "error": {"message": str(e)},
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


if __name__ == "__main__":
    print("🤖 Starting Streaming Chat with Memory Agent...")
    print("📋 Available endpoints:")
    print("  🔄 OpenAI Compatible:")
    print("    - POST /v1/chat/completions - OpenAI-compatible chat (supports stream parameter)")
    print("  🛠️ Utility Endpoints:")
    print("    - POST /callbacks/health - Health check")
    print("    - POST /callbacks/get_memories - Get user memories")
    print("    - POST /callbacks/clear_memories - Clear user memories")
    print("\n🌐 Server will run on http://0.0.0.0:8080")
    print("💡 Make sure FRIENDLI_TOKEN environment variable is set!")
    print("🔗 Use as OpenAI base_url: http://localhost:8080/v1")

    app.run(host="0.0.0.0", port=8080)
