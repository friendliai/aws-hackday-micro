"""Friendli Agent SDK."""

import inspect
import json
from collections.abc import Callable, Mapping
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, Response, StreamingResponse


class AgentApp(FastAPI):
    """FastAPI-based application for managing and executing agent callbacks.

    This class provides a simple interface for registering functions as callbacks
    that can be invoked via HTTP POST requests. It supports both synchronous and
    asynchronous functions, as well as streaming responses.
    """

    def __init__(self) -> None:
        """Initialize the AgentApp with an empty callback registry."""
        super().__init__()
        self.registered_callbacks: dict[str, Callable[..., Any]] = {}
        self.add_api_route(
            "/callbacks/{callback_name}",
            self.execute_callback,
            methods=["POST"],
        )

    def callback(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to register a function as a callback.

        This decorator registers the function in the callback registry,
        making it accessible via HTTP POST requests to /callbacks/{function_name}.

        Args:
            func: The function to register as a callback

        Returns:
            The original function unchanged (for decorator chaining)

        Example:
            @app.callback
            def my_function(param: str) -> dict:
                return {"result": param}
        """
        callback_name: str = getattr(func, "__name__", repr(func))
        self.registered_callbacks[callback_name] = func
        return func

    async def execute_callback(
        self,
        callback_name: str,
        callback_arguments: Mapping[str, Any] = {},
    ) -> Response:
        """Execute a registered callback by name.

        This method looks up a callback by name and executes it with the provided
        arguments. It handles both synchronous and asynchronous functions, as well
        as generator-based streaming responses.

        Args:
            callback_name: The name of the callback to execute
            callback_arguments: The keyword arguments to pass to the callback

        Returns:
            JSONResponse for regular results or StreamingResponse for generators

        Raises:
            HTTPException: 404 if callback not found, 500 for execution errors
        """
        # Retrieve the registered callback function
        target_callback: Callable[..., Any] | None = self.registered_callbacks.get(callback_name)
        if not target_callback:
            available_callbacks = list(self.registered_callbacks.keys())
            raise HTTPException(
                status_code=404,
                detail=f"Callback '{callback_name}' not found. Available callbacks: {available_callbacks}",
            )

        # Execute the callback function (async or sync)
        try:
            is_async_function: bool = inspect.iscoroutinefunction(target_callback)
            execution_result: Any = (
                await target_callback(**callback_arguments)
                if is_async_function
                else await run_in_threadpool(target_callback, **callback_arguments)
            )
        except TypeError as type_error:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid arguments for callback '{callback_name}': {type_error!s}",
            ) from type_error
        except Exception as execution_error:
            raise HTTPException(
                status_code=500,
                detail=f"Error executing callback '{callback_name}': {execution_error!s}",
            ) from execution_error

        # Check if result is a generator and handle streaming
        is_sync_generator: bool = inspect.isgenerator(execution_result)
        is_async_generator: bool = inspect.isasyncgen(execution_result)

        if is_sync_generator or is_async_generator:
            try:
                if is_sync_generator:
                    return StreamingResponse(
                        content=(
                            self._encode_server_sent_event(chunk) for chunk in execution_result
                        ),
                        media_type="text/event-stream",
                    )
                else:  # is_async_generator
                    return StreamingResponse(
                        content=(
                            self._encode_server_sent_event(chunk)
                            async for chunk in execution_result
                        ),
                        media_type="text/event-stream",
                    )
            except Exception as streaming_error:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error during streaming response: {streaming_error!s}",
                ) from streaming_error

        # Return regular (non-streaming) JSON response
        return JSONResponse(content=execution_result)

    def _encode_server_sent_event(self, event_data: Any) -> bytes:
        """Encode data as a Server-Sent Event (SSE) message.

        Server-Sent Events use a specific text format where each message
        is prefixed with 'data: ' and terminated with double newlines.

        Args:
            event_data: The data to encode as SSE (must be JSON-serializable)

        Returns:
            The encoded SSE message as UTF-8 bytes

        Raises:
            TypeError: If event_data is not JSON-serializable
        """
        try:
            json_encoded_data: str = json.dumps(event_data)
            sse_formatted_message: str = f"data: {json_encoded_data}\n\n"
            return sse_formatted_message.encode("utf-8")
        except (TypeError, ValueError) as encoding_error:
            # Re-raise with more context
            raise TypeError(
                f"Cannot encode data as SSE: {event_data} is not JSON-serializable"
            ) from encoding_error

    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        **uvicorn_kwargs: Any,
    ) -> None:
        """Start the FastAPI server using Uvicorn.

        Args:
            host: The host address to bind to (default: 0.0.0.0 for all interfaces)
            port: The port to bind to (default: 8080)
            **uvicorn_kwargs: Additional keyword arguments to pass to uvicorn.run()

        Example:
            app.run(host="localhost", port=8000, reload=True)
        """
        uvicorn.run(self, host=host, port=port, **uvicorn_kwargs)
