"""Example 9: Asynchronous execution for long running agents with LLM streaming + CrewAI."""

import asyncio
import json
import os
import uuid
from datetime import UTC, datetime
from typing import Any, ClassVar, Literal

from crewai import LLM, Agent, Crew, Process, Task
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

from friendli_app.sdk import AgentApp

app = AgentApp()

# Initialize OpenAI client for Friendli serverless endpoints
client = OpenAI(
    api_key=os.getenv("FRIENDLI_TOKEN"),
    base_url="https://api.friendli.ai/serverless/v1",
)

# Default model to use
DEFAULT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
DEFAULT_PORT = 8080

# Global storage for task statuses
task_storage: dict[str, dict[str, Any]] = {}


# OpenAI-compatible Pydantic models
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: dict[str, Any]


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: list[ChatMessage]
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = "auto"
    max_tokens: int | None = 1000
    temperature: float | None = 0.7
    stream: bool | None = False


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


# ---- User-facing formatting helpers ----
def _should_enable_tools(messages: list[dict[str, Any]]) -> bool:
    """Check if tools should be enabled based on the last user message."""
    last_user_msg = next(
        (
            msg.get("content", "")
            for msg in reversed(messages)
            if msg.get("role") == "user" and isinstance(msg.get("content"), str)
        ),
        "",
    )

    if not last_user_msg:
        return False

    text = last_user_msg.lower()
    keywords = [
        "research",
        "analyze",
        "investigate",
        "report",
        "write",
        "summary",
        "status",
        "progress",
        "result",
        "task",
        "start",
        "crew",
        "background",
    ]

    return any(k in text for k in keywords) or len(text) >= 80


class MessageFormatter:
    """Centralized message formatting to avoid duplication."""

    @staticmethod
    def format_tool_result(func_name: str, result: dict[str, Any]) -> str:
        """Format tool execution result based on function name."""
        formatters = {
            "start_crewai_task": MessageFormatter._format_start_task,
            "get_task_status": MessageFormatter._format_status,
            "list_all_tasks": MessageFormatter._format_list_tasks,
            "get_task_result": MessageFormatter._format_result,
        }

        formatter = formatters.get(func_name)
        if formatter:
            return formatter(result)
        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _format_start_task(result: dict[str, Any]) -> str:
        task_id = result.get("task_id", "-")
        return (
            f"Internal research has started. Task ID: {task_id}\n"
            "Ask 'How far along is it?' or \"status: TASK_ID\" to check progress."
        )

    @staticmethod
    def _format_status(task: dict[str, Any]) -> str:
        status = task.get("status", "unknown")
        progress = task.get("progress", 0)
        task_id = task.get("id", task.get("task_id", "-"))

        lines = [f"Task ID: {task_id}", f"Status: {status}", f"Progress: {progress}%"]

        if status == "completed":
            lines.append("Results are ready. Ask 'show results' or \"result: TASK_ID\".")

        if task.get("error"):
            lines.append(f"Error: {task['error']}")

        return "\n".join(lines)

    @staticmethod
    def _format_result(task: dict[str, Any]) -> str:
        task_id = task.get("task_id") or task.get("id")
        result = task.get("result", "No result available")
        return f"Task ID: {task_id}\nResearch result:\n\n{result}"

    @staticmethod
    def _format_list_tasks(result: dict[str, Any]) -> str:
        tasks = result.get("tasks", [])
        total = result.get("total_tasks", 0)

        lines = [f"Total {total} tasks"]
        for t in tasks:
            lines.append(
                f"- {t.get('id')} | status: {t.get('status')} | "
                f"progress: {t.get('progress', 0)}% | created: {t.get('created_at')}"
            )
        return "\n".join(lines)


# Function definitions for CrewAI tools
CREWAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "start_crewai_task",
            "description": "Start a new CrewAI task for comprehensive research and analysis. This creates a multi-agent workflow that runs in the background.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "Detailed description of the task to be analyzed by the CrewAI agents",
                    }
                },
                "required": ["task_description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_task_status",
            "description": "Get the current status of a CrewAI task. Returns status, progress, and results if completed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The unique identifier of the task to check",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_all_tasks",
            "description": "List all CrewAI tasks and their current statuses. Useful for getting an overview of all running and completed tasks.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_task_result",
            "description": "Get the detailed results of a completed CrewAI task. Only works for tasks with 'completed' status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The unique identifier of the completed task",
                    }
                },
            },
        },
    },
]


class CrewAIAgent:
    """CrewAI integration for background processing."""

    def __init__(self):
        self.friendli_token = os.getenv("FRIENDLI_TOKEN")

    def create_crew(self, task_description: str) -> Crew:
        """Create a CrewAI crew for the given task."""

        llm = LLM(
            model=DEFAULT_MODEL,
            api_key=self.friendli_token,
            base_url="https://api.friendli.ai/serverless/v1",
        )

        # Create agents
        researcher = Agent(
            role="Senior Research Analyst",
            goal="Conduct comprehensive research and analysis",
            backstory="""You are a senior research analyst with extensive experience
            in gathering, analyzing, and synthesizing complex information from various sources.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

        writer = Agent(
            role="Content Writer",
            goal="Create engaging and informative content",
            backstory="""You are a skilled content writer who specializes in
            transforming complex research into clear, engaging, and actionable content.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

        # Create tasks
        research_task = Task(
            description=f"""
            Research the following topic thoroughly: {task_description}

            Your analysis should include:
            1. Key concepts and definitions
            2. Current trends and developments
            3. Potential challenges and opportunities
            4. Relevant statistics or data points

            Provide a comprehensive research summary.
            """,
            expected_output="A detailed research report with key findings and insights",
            agent=researcher,
        )

        writing_task = Task(
            description="""
            Based on the research findings, create a well-structured report that:
            1. Summarizes the key points clearly
            2. Highlights the most important insights
            3. Provides actionable recommendations
            4. Is written in an engaging, professional tone

            The report should be comprehensive yet accessible.
            """,
            expected_output="A polished, professional report based on the research findings",
            agent=writer,
            context=[research_task],
        )

        # Create and return crew
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            process=Process.sequential,
            verbose=True,
        )

        return crew


async def run_crewai_background(task_id: str, task_description: str):
    """Run CrewAI crew in background and update task status."""

    def update_task(updates: dict[str, Any]):
        """Helper to update task with timestamp."""
        task_storage[task_id].update(updates)
        task_storage[task_id]["updated_at"] = datetime.now(UTC).isoformat()

    try:
        # Mark as running
        update_task({"status": "running", "progress": 10})

        # Progress updater task
        async def update_progress():
            """Periodically update progress until task completes."""
            while task_storage.get(task_id, {}).get("status") == "running":
                current = task_storage[task_id].get("progress", 0)
                if current < 90:
                    update_task({"progress": min(90, current + 10)})
                await asyncio.sleep(3.0)

        # Start progress updater
        progress_task = asyncio.create_task(update_progress())

        try:
            # Create and run crew (blocking operation)
            crewai_agent = CrewAIAgent()
            crew = crewai_agent.create_crew(task_description)

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, crew.kickoff)

            # Success
            update_task(
                {
                    "status": "completed",
                    "result": str(result),
                    "progress": 100,
                    "completed_at": datetime.now(UTC).isoformat(),
                }
            )

        finally:
            progress_task.cancel()

    except Exception as e:
        # Failure
        update_task({"status": "failed", "error": str(e)})


@app.callback
async def start_crewai_task(task_description: str):
    """Start a CrewAI task in the background."""
    if not os.getenv("FRIENDLI_TOKEN"):
        return {"error": "FRIENDLI_TOKEN environment variable is not set"}

    if not task_description:
        return {"error": "task_description is required"}

    # Generate unique task ID
    task_id = str(uuid.uuid4())

    # Initialize task storage
    task_storage[task_id] = {
        "id": task_id,
        "description": task_description,
        "status": "initializing",
        "progress": 0,
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
    }

    # Start background task
    _ = asyncio.create_task(run_crewai_background(task_id, task_description))  # noqa: RUF006

    return {
        "task_id": task_id,
        "status": "started",
        "message": "CrewAI task has been started in the background",
        "created_at": datetime.now(UTC).isoformat(),
    }


@app.callback
def get_task_status(task_id: str):
    """Get the status of a CrewAI task."""
    # If no task_id provided, return most recent task if available
    if not task_id:
        if not task_storage:
            return {"error": "No tasks available"}
        latest = max(task_storage.values(), key=lambda t: t.get("created_at", ""))
        return latest

    if task_id not in task_storage:
        return {"error": f"Task {task_id} not found"}

    return task_storage[task_id]


@app.callback
def list_tasks():
    """List all tasks and their statuses."""
    return {
        "tasks": list(task_storage.values()),
        "total_tasks": len(task_storage),
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.callback
def get_task_result(task_id: str):
    """Get the detailed results of a completed CrewAI task."""
    # If no task_id provided, use latest completed task
    if not task_id:
        completed = [t for t in task_storage.values() if t.get("status") == "completed"]
        if not completed:
            return {"error": "No completed tasks found"}
        task = max(completed, key=lambda t: t.get("completed_at", ""))
        task_id = task.get("id")

    if task_id not in task_storage:
        return {"error": f"Task {task_id} not found"}

    task = task_storage[task_id]
    if task["status"] != "completed":
        return {"error": f"Task {task_id} is not completed yet. Current status: {task['status']}"}

    return {
        "task_id": task_id,
        "status": task["status"],
        "result": task.get("result", "No result available"),
        "completed_at": task.get("completed_at"),
        "created_at": task.get("created_at"),
    }


def create_openai_response(
    completion_id: str,
    model: str,
    content: str,
    finish_reason: str = "stop",
    tool_calls: list[dict[str, Any]] | None = None,
) -> ChatCompletionResponse:
    """Create OpenAI-compatible response."""
    message = ChatMessage(role="assistant", content=content)
    if tool_calls:
        message.tool_calls = tool_calls

    return ChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=int(datetime.now(UTC).timestamp()),
        model=model,
        choices=[ChatCompletionChoice(index=0, message=message, finish_reason=finish_reason)],
        usage=ChatCompletionUsage(
            prompt_tokens=0,  # Simplified for now
            completion_tokens=0,
            total_tokens=0,
        ),
    )


def create_stream_chunk(
    completion_id: str,
    model: str,
    content: str = "",
    finish_reason: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> str:
    """Create streaming chunk in OpenAI format."""
    delta = {}
    if content:
        delta["content"] = content
    if tool_calls:
        delta["tool_calls"] = tool_calls

    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(datetime.now(UTC).timestamp()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


async def execute_function_call(function_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a function call and return the result."""
    try:
        # Map function names to app callbacks
        function_map = {
            "start_crewai_task": lambda: start_crewai_task(arguments.get("task_description", "")),
            "get_task_status": lambda: get_task_status(arguments.get("task_id", "")),
            "list_all_tasks": lambda: list_tasks(),
            "get_task_result": lambda: get_task_result(arguments.get("task_id", "")),
        }

        func = function_map.get(function_name)
        if not func:
            return {"error": f"Unknown function: {function_name}"}

        # Execute the function
        result = func()

        # Handle async functions
        if asyncio.iscoroutine(result):
            result = await result

        return result

    except Exception as e:
        return {"error": f"Function execution failed: {e!s}"}


class ChatCompletionHandler:
    """Handles chat completion requests with cleaner separation of concerns."""

    SYSTEM_MESSAGE: ClassVar[dict[str, str]] = {
        "role": "system",
        "content": (
            "You are a helpful assistant that can kick off background research, "
            "periodically update progress, and return results when ready. "
            "When the user asks to start research, confirm that internal research has started "
            "and provide a short task ID. When asked about progress, summarize current status and a rough percent. "
            "When complete, present the final findings. Use English only."
        ),
    }

    def __init__(self, request: ChatCompletionRequest):
        self.request = request
        self.completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        self.messages = self._prepare_messages()
        self.tools = request.tools or CREWAI_TOOLS

    def _prepare_messages(self) -> list[dict]:
        """Prepare messages with system message and filtering."""
        messages = [self.SYSTEM_MESSAGE]

        # Add non-system messages
        for msg in self.request.messages:
            if msg.role != "system":
                messages.append({"role": msg.role, "content": msg.content})

        return messages

    async def handle_streaming(self) -> StreamingResponse:
        """Handle streaming response."""
        if _should_enable_tools(self.messages):
            generator = stream_openai_response_with_tools(
                self.completion_id, self.request.model, self.messages, self.tools, self.request
            )
        else:
            generator = stream_simple_response(
                self.completion_id, self.request.model, self.messages, self.request
            )

        return StreamingResponse(generator, media_type="text/event-stream")

    async def handle_non_streaming(self) -> ChatCompletionResponse:
        """Handle non-streaming response."""
        # Determine if tools should be used
        use_tools = _should_enable_tools(self.messages)

        # Create completion request
        response = client.chat.completions.create(
            model=self.request.model,
            messages=self.messages,
            tools=self.tools if use_tools else None,
            tool_choice=self.request.tool_choice if use_tools else None,
            max_tokens=self.request.max_tokens,
            temperature=self.request.temperature,
            stream=False,
        )

        # Handle function calls if present
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                result = await execute_function_call(func_name, func_args)
                result_text = MessageFormatter.format_tool_result(func_name, result)
                return create_openai_response(self.completion_id, self.request.model, result_text)

        # Regular response
        content = response.choices[0].message.content
        return create_openai_response(self.completion_id, self.request.model, content)


# Add OpenAI-compatible endpoint
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> Any:
    """
    OpenAI-compatible chat completions endpoint with CrewAI function calling.
    Supports both streaming and non-streaming based on the 'stream' parameter.
    """
    # Validation
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    if not os.getenv("FRIENDLI_TOKEN"):
        raise HTTPException(
            status_code=500, detail="FRIENDLI_TOKEN environment variable is not set"
        )

    try:
        handler = ChatCompletionHandler(request)

        if request.stream:
            return await handler.handle_streaming()
        else:
            return await handler.handle_non_streaming()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


async def stream_openai_response_with_tools(
    completion_id: str,
    model: str,
    messages: list[dict],
    tools: list[dict[str, Any]],
    request: ChatCompletionRequest,
):
    """Generate streaming response with tool support in OpenAI format."""
    try:
        # Start streaming
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=request.tool_choice,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=True,
        )

        # Accumulate tool calls properly
        accumulated_tool_calls = {}

        for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue

            # Handle tool calls - accumulate them internally; DO NOT expose to client
            if choice.delta.tool_calls:
                for tool_call_delta in choice.delta.tool_calls:
                    # Get index (this is crucial for OpenAI tool calls)
                    index = getattr(tool_call_delta, "index", 0)

                    # Initialize this tool call if not exists
                    if index not in accumulated_tool_calls:
                        accumulated_tool_calls[index] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }

                    # Update ID if present
                    if hasattr(tool_call_delta, "id") and tool_call_delta.id:
                        accumulated_tool_calls[index]["id"] = tool_call_delta.id

                    # Update function details if present
                    if hasattr(tool_call_delta, "function") and tool_call_delta.function:
                        if (
                            hasattr(tool_call_delta.function, "name")
                            and tool_call_delta.function.name
                        ):
                            accumulated_tool_calls[index]["function"]["name"] = (
                                tool_call_delta.function.name
                            )
                        if (
                            hasattr(tool_call_delta.function, "arguments")
                            and tool_call_delta.function.arguments is not None
                        ):
                            accumulated_tool_calls[index]["function"]["arguments"] += (
                                tool_call_delta.function.arguments
                            )

                    # Do not yield tool delta to end user; keep quiet

            # Handle regular content
            elif choice.delta.content:
                content = choice.delta.content
                yield create_stream_chunk(completion_id, model, content)

        # Execute completed tool calls
        if accumulated_tool_calls:
            for tool_call_info in accumulated_tool_calls.values():
                func_name = tool_call_info["function"]["name"]
                func_args_str = tool_call_info["function"]["arguments"]

                if func_name and func_args_str:
                    try:
                        func_args = json.loads(func_args_str)
                        result = await execute_function_call(func_name, func_args)

                        # Convert to user-facing text
                        text = MessageFormatter.format_tool_result(func_name, result)
                        yield create_stream_chunk(completion_id, model, text)

                    except json.JSONDecodeError as e:
                        error_text = (
                            f"\n\n❌ Function '{func_name}' failed: Invalid JSON arguments: {e!s}"
                        )
                        yield create_stream_chunk(completion_id, model, error_text)
                    except Exception as e:
                        error_text = f"\n\n❌ Function '{func_name}' failed: {e!s}"
                        yield create_stream_chunk(completion_id, model, error_text)

        # Final chunk with finish_reason
        yield create_stream_chunk(completion_id, model, finish_reason="stop")

        # End stream
        yield "data: [DONE]\n\n"

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


async def stream_simple_response(
    completion_id: str, model: str, messages: list[dict], request: ChatCompletionRequest
):
    """Generate simple streaming response without tools."""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=True,
        )

        for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue

            if choice.delta.content:
                content = choice.delta.content
                yield create_stream_chunk(completion_id, model, content)

        # Final chunk with finish_reason
        yield create_stream_chunk(completion_id, model, finish_reason="stop")
        yield "data: [DONE]\n\n"

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


@app.callback
def ping():
    """Health check endpoint following AWS AgentCore pattern."""
    return {
        "status": "Healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "active_tasks": len([t for t in task_storage.values() if t["status"] == "running"]),
        "total_tasks": len(task_storage),
    }


@app.callback
def root():
    """Root endpoint with API documentation."""
    return {
        "name": "Async CrewAI Agent",
        "description": "OpenAI-compatible chat completion API with CrewAI multi-agent function calling",
        "version": "2.0",
        "openai_compatible_endpoint": "/v1/chat/completions",
        "function_tools": [
            "start_crewai_task - Start comprehensive research tasks",
            "get_task_status - Check task progress",
            "list_all_tasks - Overview of all tasks",
            "get_task_result - Get completed task results",
        ],
        "management_endpoints": {
            "start_crewai_task": "Start a CrewAI task in background",
            "get_task_status": "Get status of a specific task",
            "get_task_result": "Get results of completed task",
            "list_tasks": "List all tasks and their statuses",
            "ping": "Health check endpoint",
        },
        "environment_required": ["FRIENDLI_TOKEN"],
        "timestamp": datetime.now(UTC).isoformat(),
    }


if __name__ == "__main__":
    print("🤖 Starting Async CrewAI Agent with OpenAI-compatible API...")
    print(f"🌐 Server will run on http://0.0.0.0:{DEFAULT_PORT}")
    print()
    print("🔑 Required Environment Variables:")
    print("  - FRIENDLI_TOKEN: Get your token from https://friendli.ai")
    print()
    print("📋 OpenAI-Compatible API:")
    print("  - POST /v1/chat/completions - OpenAI-compatible chat with function calling")
    print("    * Supports streaming (stream=true)")
    print("    * Includes CrewAI function tools by default")
    print(f"    * Use as base_url: http://localhost:{DEFAULT_PORT}/v1")
    print()
    print("🛠️ Available Function Tools:")
    print("  - start_crewai_task: Start multi-agent research workflows")
    print("  - get_task_status: Check running task progress")
    print("  - list_all_tasks: Get overview of all tasks")
    print("  - get_task_result: Retrieve completed task results")
    print()
    print("📋 CrewAI Management API:")
    print("  - POST /callbacks/start_crewai_task - Start CrewAI task")
    print("  - POST /callbacks/get_task_status - Get task status")
    print("  - POST /callbacks/get_task_result - Get task results")
    print("  - POST /callbacks/list_tasks - List all tasks")
    print("  - POST /callbacks/ping - Health check")
    print()
    print("💡 Usage Examples:")
    print(f"  curl -X POST http://localhost:{DEFAULT_PORT}/v1/chat/completions \\")
    print('    -H "Content-Type: application/json" \\')
    print(
        '    -d \'{"model":"Qwen/Qwen3-235B-A22B-Instruct-2507","messages":[{"role":"user","content":"Analyze the future of renewable energy"}],"stream":true}\''
    )
    print()

    # Check if environment token is set
    if not os.getenv("FRIENDLI_TOKEN"):
        print("⚠️  WARNING: FRIENDLI_TOKEN environment variable is not set!")
        print("   Set it with: export FRIENDLI_TOKEN='your_token_here'")
        print()

    app.run(host="0.0.0.0", port=DEFAULT_PORT)
