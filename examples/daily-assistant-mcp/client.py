import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main():
    mcp_url = "http://localhost:8080/mcp"
    headers = {}

    async with (
        streamablehttp_client(mcp_url, headers, timeout=120, terminate_on_close=False) as (
            read_stream,
            write_stream,
            _,
        ),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        tools = await session.list_tools()
        print("Available tools:", [t.name for t in tools.tools])

        # Test some tools
        # 1. calculate_tip
        print("\n--- Testing calculate_tip ---")
        result = await session.call_tool("calculate_tip", {"amount": 50.0, "tip_percentage": 18})
        print(f"Result: {result}")

        # 2. generate_password
        print("\n--- Testing generate_password ---")
        result = await session.call_tool("generate_password", {"length": 16})
        print(f"Result: {result}")

        # 3. convert_timezone
        print("\n--- Testing convert_timezone ---")
        result = await session.call_tool(
            "convert_timezone",
            {
                "time_str": "2024-12-20 15:00",
                "from_timezone": "America/New_York",
                "to_timezone": "Asia/Tokyo",
            },
        )
        print(f"Result: {result}")

        # 4. calculate_bmi
        print("\n--- Testing calculate_bmi ---")
        result = await session.call_tool("calculate_bmi", {"weight_kg": 70, "height_cm": 175})
        print(f"Result: {result}")

        # 5. unit_converter
        print("\n--- Testing unit_converter ---")
        result = await session.call_tool(
            "unit_converter", {"value": 100, "from_unit": "kg", "to_unit": "lbs"}
        )
        print(f"Result: {result}")

        # 6. text_statistics
        print("\n--- Testing text_statistics ---")
        result = await session.call_tool(
            "text_statistics",
            {"text": "This is a simple test message to analyze. It contains multiple sentences."},
        )
        print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
